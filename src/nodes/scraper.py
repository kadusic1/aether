from langchain_core.messages import HumanMessage, SystemMessage

from niche_config.common import scraper_prompt
from src.models import load_chat_model
from src.state import VideoState
from src.tools import scrape_url


async def scraper(state: VideoState) -> dict:
    """
    Scrape URLs from sources and generate a structured
    overview using the LLM.

    Defense in depth: early-returns with an empty
    overview if no sources exist (the workflow also
    gates this node via a conditional edge, but the
    node handles it gracefully regardless).

    Stores the result in sources_overview only, not
    in messages.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'sources_overview' containing the
        LLM-generated summary of scraped content.
    """
    sources = state.get("sources", [])
    if not sources:
        return {"sources_overview": ""}

    # Scrape all URLs using the existing tool
    raw_content = await scrape_url(sources)

    # Generate structured overview via LLM
    model = load_chat_model()
    response = await model.ainvoke(
        [
            SystemMessage(
                content=scraper_prompt.format(
                    persona=state["niche"].persona,
                ),
            ),
            HumanMessage(content=raw_content),
        ],
        reasoning=True,
    )
    return {"sources_overview": response}
