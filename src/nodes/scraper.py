from langchain_core.messages import HumanMessage, SystemMessage

from niche_config.common import scraper_prompt
from src.models import load_chat_model
from src.state import VideoState
from src.tools import scrape_url, transcript_youtube_videos
import asyncio
from src.utils import empty


async def scraper(state: VideoState) -> dict:
    """
    Scrape URLs from sources and generate a structured
    overview using the LLM.

    Defense in depth: early-returns with an empty
    overview if no sources exist (the workflow also
    gates this node via a conditional edge, but the
    node handles it gracefully regardless).

    Web scraping and transcript fetching run
    concurrently via asyncio.gather. The synchronous
    transcript_youtube_videos call is offloaded to a
    thread.

    Stores the result in sources_overview only, not
    in messages.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'sources_overview' containing the
        LLM-generated summary of scraped content.
    """
    web_sources = state.get("web_sources", [])
    yt_sources = state.get("youtube_sources", [])

    if not web_sources and not yt_sources:
        return {"sources_overview": ""}

    # Scrape web and fetch transcripts concurrently
    web_content, yt_content = await asyncio.gather(
        scrape_url(web_sources) if web_sources else empty(),
        asyncio.to_thread(transcript_youtube_videos, yt_sources)
        if yt_sources
        else empty(),
    )

    raw_content = web_content + yt_content

    # No need to do the LLM overview if the intent is just chat.
    # The LLM overview is focused on the "creator" side of things.
    if state["intent"] == "chat":
        return {"sources_overview": raw_content}

    # Generate structured overview via LLM
    model = load_chat_model(
        provider="mistral",
        temperature=0.65,
    )
    response = await model.ainvoke(
        [
            SystemMessage(
                content=scraper_prompt.format(
                    persona=state["niche"].persona,
                ),
            ),
            HumanMessage(content=raw_content),
        ],
    )
    return {"sources_overview": response}
