from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from niche_config.common import extract_links_prompt
from src.models import load_chat_model
from src.state import VideoState


class ExtractedLinks(BaseModel):
    """
    Structured output for URL extraction.

    Attributes:
        urls: List of URLs and bare domains found
            in the user's message. Empty if none
            found.
    """

    urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs and bare domains extracted from"
            " the text. Empty list if none found."
        ),
    )


async def extract_links(state: VideoState) -> dict:
    """
    Extract URLs from the latest user message using
    the LLM with structured output.

    Handles both full URLs (https://...) and bare
    domains (vt.tiktok.com/abc, reddit.com/r/topic).

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'sources' key containing extracted
        URLs (merged via operator.add reducer).
    """
    last_message = state["messages"][-1]
    model = load_chat_model().with_structured_output(
        ExtractedLinks,
    )
    response = await model.ainvoke(
        [
            SystemMessage(
                content=extract_links_prompt.format(
                    persona=state["niche"].persona,
                )
            ),
            HumanMessage(content=last_message.content),
        ],
        reasoning=False,
    )
    return {"sources": response.urls}
