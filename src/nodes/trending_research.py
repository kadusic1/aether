from src.state import VideoState
from src.services import get_searx
import asyncio


async def trending_research(state: VideoState) -> VideoState:
    """
    Rearch trending topics in the niche using Searx and update the state with the results.
    search_results is a list of lists, where each inner list contains the
    search results for a specific query.
    The fields of each search result include:
    - title: The title of the search result.
    - snippet: A brief snippet or description of the search result.
    - link: The URL link to the search result.
    - engines: A list of search engines that returned this result.
    - category: The category of the search result (e.g., "general", "news", "images").

    Args:
        state (VideoState): The current state of the video creation process.
    Returns:
        VideoState: The updated state with trending topics research results.
    """
    searx = get_searx()
    search_results = await asyncio.gather(*[
        searx.aresults(query=query, num_results=10, time_range="day")
        for query in state["niche"].trending_search_queries
    ])
    return {
        "trending_topics": search_results,
    }
