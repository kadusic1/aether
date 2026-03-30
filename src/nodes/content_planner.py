from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from niche_config.common import content_planner_prompt
from src.models import load_chat_model
from src.schemas import ContentPlan
from src.state import VideoState
from src.utils import combine_source_and_user_prompt


async def content_planner(state: VideoState) -> dict:
    """
    Generate a complete video production plan from
    research.

    Consumes the sources_overview and the
    user's original message to produce a ContentPlan.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'content_plan' containing the
        structured ContentPlan, and 'messages'
        with a summary for the user.
    """
    user_prompt_with_sources = combine_source_and_user_prompt(state)

    model = load_chat_model(
        provider="mistral",
        temperature=0.7,
    ).with_structured_output(ContentPlan)

    plan = await model.ainvoke(
        [
            SystemMessage(
                content=content_planner_prompt.format(
                    persona=state["niche"].persona,
                ),
            ),
            HumanMessage(content=user_prompt_with_sources),
        ],
    )

    return {
        "content_plan": plan,
    }
