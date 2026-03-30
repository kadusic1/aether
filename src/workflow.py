from langgraph.graph import END, StateGraph

from src.nodes import (
    content_planner,
    message_analyzer,
    scraper,
    search_node,
    chat,
)
from src.state import VideoState


def route_after_analysis(state: VideoState) -> str:
    """
    Conditional edge after message_analyzer.

    Routes based on search need and available sources:
    - search_node: when the LLM decided a web search
      is needed.
    - scraper: when no search is needed but URLs were
      extracted from the user's message.
    - chat: when the intent is basic_chat.
    - END: when neither search nor scraping is needed.

    Args:
        state: The current workflow state.

    Returns:
        The name of the next node, or END.
    """
    if state["intent"] == "basic_chat":
        return "chat"
    if state["use_search"]:
        return "search_node"
    if state.get("web_sources") or state.get("youtube_sources"):
        return "scraper"
    return END


def build_workflow():
    """
    Construct and compile the LangGraph workflow.

    The graph has five nodes:
    1. message_analyzer — extracts URLs, decides
       search, and classifies intent in one LLM call.
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
        route_after_analysis,
        {
            "search_node": "search_node",
            "scraper": "scraper",
            "chat": "chat",
            END: END,
        },
    )

    # Chat node terminates the workflow
    graph_builder.add_edge("chat", END)

    # Search results always flow to scraper
    graph_builder.add_edge("search_node", "scraper")

    # Scraper flows into content planner
    graph_builder.add_edge("scraper", "content_planner")

    # Content planner terminates the workflow
    graph_builder.add_edge("content_planner", END)

    return graph_builder.compile()
