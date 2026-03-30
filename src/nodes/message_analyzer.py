from langchain_core.messages import HumanMessage, SystemMessage

from niche_config.common import message_analyzer_prompt
from src.models import load_chat_model
from src.state import VideoState
from src.schemas import MessageAnalysis


async def message_analyzer(state: VideoState) -> dict:
    """
    Analyze the user's message in a single LLM call.

    Parses the user's intent from the command prefix
    and performs URL extraction and search routing
    using structured output.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'web_sources', 'youtube_sources',
        'use_search', and 'intent'.
    """
    last_message = state["messages"][-1]

    # 1. Parse intent via slash command
    if last_message.content.startswith("/chat"):
        intent = "chat"
    elif last_message.content.startswith("/plan"):
        intent = "plan"
    elif last_message.content.startswith("/generate"):
        intent = "generate"
    else:
        raise ValueError("Unrecognized command prefix in user message")

    model = load_chat_model(
        temperature=0.0,
        provider="google",
    ).with_structured_output(
        MessageAnalysis,
    )

    # Remove first word (the command) before sending the prompt to the model.
    _, _, prompt = last_message.content.partition(" ")

    response = await model.ainvoke(
        [
            SystemMessage(
                content=message_analyzer_prompt.format(
                    persona=state["niche"].persona,
                ),
            ),
            HumanMessage(content=prompt),
        ],
        reasoning=False,
    )
    return {
        "web_sources": response.web_urls,
        "youtube_sources": response.yt_ids,
        "use_search": response.use_search,
        "intent": intent,
    }
