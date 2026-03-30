import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from niche_config import Niche
from src.schemas import ContentPlan


class VideoState(TypedDict):
    """
    State definition for the video creation workflow.

    Attributes:
        niche: The niche configuration for the current
            video.
        messages: Conversation history, automatically
            appended by LangGraph.
        web_sources: Accumulated web URLs from user
            input and search results. Uses operator.add
            reducer so multiple nodes can append without
            overwriting.
        youtube_sources: Accumulated YouTube video IDs
            from user input and search results. Uses
            operator.add reducer.
        sources_overview: LLM-generated summary of
            scraped source content.
        content_plan: Production plan for the
            video. Set by the content_planner node.
            None until planning is complete.
        intent: Classified intent of the user's prompt
            (plan, generate or chat).
        use_search: Transient routing flag set by the
            search router node.
    """

    niche: Niche
    messages: Annotated[list[AnyMessage], add_messages]
    web_sources: Annotated[list[str], operator.add]
    youtube_sources: Annotated[list[str], operator.add]
    sources_overview: str
    content_plan: ContentPlan | None
    intent: str
    use_search: bool
