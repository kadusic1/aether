from langgraph.graph import END, StateGraph

from src.nodes.extract_links import extract_links
from src.nodes.intent_router import intent_router
from src.nodes.scraper import scraper
from src.nodes.search_node import search_node
from src.nodes.search_router import search_router
from src.state import VideoState


def route_search(state: VideoState) -> str:
    """
    Conditional edge after search_router.

    Routes to search_node if the LLM decided search
    is needed, otherwise skips directly to scraper.

    Args:
        state: The current workflow state.

    Returns:
        The name of the next node.
    """
    if state["use_search"]:
        return "search_node"
    return "scraper"


def route_intent(state: VideoState) -> str:
    """
    Conditional edge after intent_router.

    Fans out to the appropriate downstream handler
    based on classified intent. Currently all routes
    terminate at END (placeholders for future
    sub-graphs).

    Args:
        state: The current workflow state.

    Returns:
        The classified intent string.
    """
    return state["intent"]


def build_workflow():
    """
    Construct and compile the LangGraph workflow.

    Returns:
        The compiled StateGraph ready for execution.
    """
    graph_builder = StateGraph(VideoState)

    # Nodes
    graph_builder.add_node("extract_links", extract_links)
    graph_builder.add_node("search_router", search_router)
    graph_builder.add_node("search_node", search_node)
    graph_builder.add_node("scraper", scraper)
    graph_builder.add_node("intent_router", intent_router)

    # Edges
    graph_builder.set_entry_point("extract_links")
    graph_builder.add_edge("extract_links", "search_router")

    # Search router conditional: search or skip
    graph_builder.add_conditional_edges(
        "search_router",
        route_search,
        {
            "search_node": "search_node",
            "scraper": "scraper",
        },
    )

    # Search node always flows to scraper
    graph_builder.add_edge("search_node", "scraper")

    # Scraper always flows to intent router
    graph_builder.add_edge("scraper", "intent_router")

    # Intent router fans out (all END for now)
    graph_builder.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "video_planning": END,
            "video_generation": END,
            "basic_chat": END,
        },
    )

    return graph_builder.compile()
