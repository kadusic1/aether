import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from niche_config import Niche


class VideoState(TypedDict):
    """
    State definition for the video creation workflow.

    Attributes:
        niche: The niche configuration for the current
            video.
        messages: Conversation history, automatically
            appended by LangGraph.
        sources: Accumulated URLs from user input and
            search results. Uses operator.add reducer
            so multiple nodes can append without
            overwriting.
        sources_overview: LLM-generated summary of
            scraped source content.
        intent: Classified intent of the user's prompt
            (video_planning, video_generation,
            basic_chat).
        use_search: Transient routing flag set by the
            search router node.
    """

    niche: Niche
    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[list[str], operator.add]
    sources_overview: str
    intent: str
    use_search: bool
