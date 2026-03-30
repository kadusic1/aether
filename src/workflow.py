from langgraph.graph import END, StateGraph

from src.nodes import (
    content_planner,
    message_analyzer,
    scraper,
    search_node,
    chat,
)
from src.state import VideoState


def _route_from_analyzer(state: VideoState) -> str:
    """
    Route from the analyzer node to search if needed,
    otherwise check for scraper sources. Search is not used for chat intent.

    Args:
        state: The current workflow state

    Returns:
        The name of the next node to route to.
    """
    if state.get("use_search") and state.get("intent") != "chat":
        return "search_node"
    return _route_to_scraper(state)


def _route_to_scraper(state: VideoState) -> str:
    """
    Route to scraper if there are URLs, otherwise skip
    straight to the intent node.

    Args:
        state: The current workflow state

    Returns:
        The name of the next node to route to.
    """
    if state.get("web_sources") or state.get("youtube_sources"):
        return "scraper"
    return _route_by_intent(state)


def _route_by_intent(state: VideoState) -> str:
    """
    Route to the final intent node based on the user's
    extracted command.

    Args:
        state: The current workflow state

    Returns:
        The name of the next node to route to.
    """
    intent = state.get("intent")
    if intent == "chat":
        return "chat"
    if intent == "plan":
        return "content_planner"
    # If "generate" (not implemented yet) or "invalid", terminate the graph
    return END


def build_workflow():
    """
    Construct and compile the LangGraph workflow.

    The graph has five nodes:
    1. message_analyzer — extracts URLs, decides
       search, and parses intent.
    2. search_node — generates a query, searches the
       web, and picks the best URLs.
    3. scraper — scrapes URLs and summarizes content.
    4. content_planner — transforms research into a
       frozen video production plan.
    5. chat — handles general conversation.

    Returns:
        The compiled StateGraph ready for execution.
    """
    graph_builder = StateGraph(VideoState)

    # Nodes
    graph_builder.add_node("message_analyzer", message_analyzer)
    graph_builder.add_node("search_node", search_node)
    graph_builder.add_node("scraper", scraper)
    graph_builder.add_node("content_planner", content_planner)
    graph_builder.add_node("chat", chat)

    # Edges
    graph_builder.set_entry_point("message_analyzer")

    graph_builder.add_conditional_edges(
        "message_analyzer",
        _route_from_analyzer,
        {
            "search_node": "search_node",
            "scraper": "scraper",
            "chat": "chat",
            "content_planner": "content_planner",
            END: END,
        },
    )

    # 2. Search routing
    graph_builder.add_conditional_edges(
        "search_node",
        _route_to_scraper,
        {
            "scraper": "scraper",
            "chat": "chat",
            "content_planner": "content_planner",
            END: END,
        },
    )

    # 3. Scraper routing
    graph_builder.add_conditional_edges(
        "scraper",
        _route_by_intent,
        {
            "chat": "chat",
            "content_planner": "content_planner",
            END: END,
        },
    )

    graph_builder.add_edge("chat", END)
    graph_builder.add_edge("content_planner", END)

    return graph_builder.compile()
