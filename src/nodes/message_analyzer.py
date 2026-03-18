from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from niche_config.common import message_analyzer_prompt
from src.models import load_chat_model
from src.state import VideoState


class MessageAnalysis(BaseModel):
    """
    Combined URL extraction, search routing, and
    intent classification from the user's message.

    Attributes:
        urls: List of URLs and bare domains found
            in the user's message. Empty if none
            found.
        use_search: Whether the user's message
            requires a web search.
        intent: The classified intent of the user's
            prompt.
    """

    urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs and bare domains extracted from"
            " the text. Empty list if none found."
        ),
    )
    use_search: bool = Field(
        description=(
            "True if the user's message requires a web search, False otherwise."
        ),
    )
    intent: Literal[
        "video_planning",
        "video_generation",
        "basic_chat",
    ] = Field(
        description=(
            "The classified intent: video_planning,"
            " video_generation, or basic_chat."
        ),
    )


async def message_analyzer(state: VideoState) -> dict:
    """
    Analyze the user's message in a single LLM call.

    Performs URL extraction, search routing, and intent
    classification together using structured output.
    Replaces the previously separate extract_links,
    search_router, and intent_router nodes.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'sources' (extracted URLs),
        'use_search' (boolean flag), and 'intent'
        (classified intent string).
    """
    last_message = state["messages"][-1]
    model = load_chat_model(
        temperature=0.0,
        provider="google",
    ).with_structured_output(
        MessageAnalysis,
    )
    response = await model.ainvoke(
        [
            SystemMessage(
                content=message_analyzer_prompt.format(
                    persona=state["niche"].persona,
                ),
            ),
            HumanMessage(content=last_message.content),
        ],
        reasoning=False,
    )
    return {
        "sources": response.urls,
        "use_search": response.use_search,
        "intent": response.intent,
    }
