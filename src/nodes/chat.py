from langchain_core.messages import AIMessage, SystemMessage
from niche_config.common import chat_prompt
from src.models import load_chat_model
from src.state import VideoState
from src.tools import scrape_url, transcript_youtube_videos
from src.utils import empty
import asyncio


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
    web_sources = state.get("web_sources", [])
    yt_sources = state.get("youtube_sources", [])
    system_content = chat_prompt
    if web_sources or yt_sources:
        web_content, yt_content = await asyncio.gather(
            scrape_url(web_sources) if web_sources else empty(),
            asyncio.to_thread(
                transcript_youtube_videos,
                yt_sources,
            )
            if yt_sources
            else empty(),
        )
        scraped = (web_content or "") + (yt_content or "")
        system_content += (
            "\n\n## AVAILABLE SOURCES\n"
            "Use the following content to ground "
            "your response. Cite specifics when "
            "relevant.\n\n"
            f"{scraped}"
        )
    model = load_chat_model(
        provider="google",
        temperature=0.7,
    )
    response = await model.ainvoke(
        [SystemMessage(content=system_content), state["messages"][-1]],
        reasoning=False,
    )
    return {"messages": [AIMessage(content=response)]}
