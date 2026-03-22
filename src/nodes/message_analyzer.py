from langchain_core.messages import HumanMessage, SystemMessage

from niche_config.common import message_analyzer_prompt
from src.models import load_chat_model
from src.state import VideoState
from src.schemas import MessageAnalysis


async def message_analyzer(state: VideoState) -> dict:
    """
    Analyze the user's message in a single LLM call.

    Performs URL extraction, search routing, and intent
    classification together using structured output.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'web_sources', 'youtube_sources',
        'use_search', and 'intent'.
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
        "web_sources": response.web_urls,
        "youtube_sources": response.yt_ids,
        "use_search": response.use_search,
        "intent": response.intent,
    }
