from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from niche_config.common import intent_router_prompt
from src.models import load_chat_model
from src.state import VideoState


class IntentDecision(BaseModel):
    """
    Structured output for intent classification.

    Attributes:
        intent: The classified intent of the user's
            prompt.
    """

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


async def intent_router(state: VideoState) -> dict:
    """
    Classify the user's intent into one of three
    categories.

    Operates on the fully enriched state — has access
    to the original prompt, extracted sources, and
    sources_overview. This allows for more accurate
    intent classification.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'intent' string value.
    """
    last_message = state["messages"][-1]

    # Build context including sources overview if
    # available, so the LLM has full picture.
    user_text = str(last_message.content)
    overview = state.get("sources_overview", "")
    if overview:
        full_context = f"{user_text}\n\nContext from research:\n{overview}"
    else:
        full_context = user_text

    model = load_chat_model().with_structured_output(
        IntentDecision,
    )
    response = await model.ainvoke(
        [
            SystemMessage(
                content=intent_router_prompt.format(
                    persona=state["niche"].persona,
                ),
            ),
            HumanMessage(content=full_context),
        ],
        reasoning=False,
    )
    return {"intent": response.intent}
