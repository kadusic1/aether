from langchain_core.messages import HumanMessage, SystemMessage

from niche_config.common import search_picker_prompt, search_query_prompt
from src.models import load_chat_model
from src.state import VideoState
from src.tools import search_web, search_youtube
import asyncio
from src.schemas import SearchQuery, SelectedResults


async def search_node(state: VideoState) -> dict:
    """
    Execute search and pick interesting results.

    Performs two LLM calls with concurrent tool
    execution in between:
    1. Generate a web query and a YouTube query in
       one LLM call.
    2. Execute both searches concurrently (YouTube
       runs in a thread since yt-dlp is synchronous).
    3. Evaluate combined results and pick the best
       web URLs and YouTube video IDs.

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'web_sources' and 'youtube_sources'
        (merged via operator.add reducers).
    """
    niche = state["niche"]
    last_message = state["messages"][-1]

    # --- Call 1: Generate both search queries ---
    query_prompt = search_query_prompt.format(
        persona=niche.persona,
        examples="\n".join(niche.trending_search_queries),
    )
    query_model = load_chat_model(
        temperature=1,
        provider="mistral",
    ).with_structured_output(
        SearchQuery,
    )
    query_result = await query_model.ainvoke(
        [
            SystemMessage(content=query_prompt),
            HumanMessage(content=last_message.content),
        ],
        reasoning=False,
    )

    # --- Execute both searches concurrently ---
    raw_results, yt_results = await asyncio.gather(
        search_web(
            query=query_result.query,
            num_results=10,
            time_range=None,
        ),
        asyncio.to_thread(
            search_youtube,
            query=query_result.youtube_query,
            num_results=15,
        ),
    )

    # Format YouTube results for the picker LLM
    yt_text = "\n".join(
        f"- {v['title']} | {v['channel']}"
        f" | {v['view_count']} views"
        f" | {v['duration']}s | ID: {v['id']}"
        for v in yt_results
    )

    combined_input = (
        "## Web Search Results\n"
        f"{raw_results}\n\n"
        "## YouTube Search Results\n"
        f"{yt_text}\n"
    )

    # --- Call 2: Pick interesting results ---
    picker_model = load_chat_model(
        provider="google",
        temperature=0.7,
    ).with_structured_output(
        SelectedResults,
    )
    picker_result = await picker_model.ainvoke(
        [
            SystemMessage(
                content=search_picker_prompt.format(
                    persona=niche.persona,
                ),
            ),
            HumanMessage(content=combined_input),
        ],
        reasoning=False,
    )

    return {
        "web_sources": picker_result.web_urls,
        "youtube_sources": picker_result.youtube_ids,
    }
