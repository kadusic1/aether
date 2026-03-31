from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from niche_config.common import chat_prompt
from src.models import load_chat_model
from src.state import VideoState


async def chat(state: VideoState) -> dict:
    """
    Generate a conversational response for the
    basic_chat intent.
    When the user's message contains URLs or YouTube
    IDs, they are scraped and injected as grounding
    context. Otherwise responds from general
    knowledge.
    Args:
        state: The current workflow state.
    Returns:
        Dict with 'messages' containing the AI reply.
    """
    last_message = state["messages"][-1]
    if state.get("sources_overview"):
        chat_system_prompt = (
            chat_prompt + "\n\nSOURCES:\n" + state["sources_overview"]
        )
    else:
        chat_system_prompt = chat_prompt

    model = load_chat_model(
        provider="google",
        temperature=0.7,
    )
    response = await model.ainvoke(
        [
            SystemMessage(content=chat_system_prompt),
            HumanMessage(content=last_message.content),
        ],
        reasoning=False,
    )
    return {"messages": [AIMessage(content=response)]}
