from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from niche_config.common import search_router_prompt
from src.models import load_chat_model
from src.state import VideoState


class SearchDecision(BaseModel):
    """
    Structured output for the search routing decision.

    Attributes:
        use_search: Whether the user's message
            requires a web search.
    """

    use_search: bool = Field(
        description=(
            "True if the user's message requires a web search, False otherwise."
        ),
    )


async def search_router(state: VideoState) -> dict:
    """
    Decide whether the user's message requires a web
    search.

    Uses the niche's search_router_prompt to evaluate
    the latest user message. The decision is stored
    in state['use_search'] and consumed by a
    conditional edge in the workflow.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'use_search' boolean flag.
    """
    last_message = state["messages"][-1]
    model = load_chat_model(
        temperature=0.2,
        provider="google",
    ).with_structured_output(
        SearchDecision,
    )
    response = await model.ainvoke(
        [
            SystemMessage(
                content=search_router_prompt.format(
                    persona=state["niche"].persona,
                ),
            ),
            HumanMessage(content=last_message.content),
        ],
        reasoning=False,
    )
    return {"use_search": response.use_search}
