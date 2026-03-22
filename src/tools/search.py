from src.services import get_searx
from typing import Any, Literal
import yt_dlp
from urllib.parse import urlencode


async def search_web(
    query: str,
    num_results: int,
    time_range: Literal["day", "week", "month", "year", None],
) -> str:
    """
    Search the web for trending topics using Searx.
    Args:
        query: The search query to execute.
        num_results: The number of search results to return (max is 15).
        time_range: The time range for the search (e.g., "day", "week", "month", "year", None).
    Returns:
        A string representation of the search results including URLs.
    """
    searx = get_searx()
    search_params = {"num_results": min(num_results, 15)}
    if time_range is not None:
        search_params["time_range"] = time_range
    results = await searx.aresults(query, **search_params)
    print(f"Search results for '{query}':", flush=True)
    print("Time range:", time_range, flush=True)
    print("Num Results:", num_results, flush=True)
    return str(results)


def search_youtube(query: str, num_results: int = 25) -> list[dict[str, Any]]:
    """Search YouTube for videos matching the given query.

    Performs a popularity-sorted YouTube search using yt-dlp with English
    language bias. The function uses custom URL parameters:
    - `sp=CAMSAigB` to sort results by popularity and with subtitles included.
    - `hl=en` to set the interface language to English.
    - `gl=US` to set the country to United States.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the search results.
    """
    # Preprocess the query ensuring to avoid Hindi results
    query = f"{query} -hindi"
    print(query, flush=True)

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "playlist_items": f"1-{num_results}",
        # extractor_args can be used to set language preferences for certain extractors
        "extractor_args": {"youtube": {"lang": ["en"]}},
    }
    params = urlencode(
        {
            "search_query": query,
            # Sort by popularity and include subtitles
            "sp": "CAMSAigB",
            "hl": "en",
            "gl": "US",
        }
    )
    search_url = f"https://www.youtube.com/results?{params}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(search_url, download=False)

    return [
        {
            "channel": entry.get("channel"),
            "description": entry.get("description"),
            "title": entry.get("title"),
            "duration": entry.get("duration"),
            "url": entry.get("url"),
            "view_count": entry.get("view_count"),
            "id": entry.get("id"),
        }
        for entry in results.get("entries", [])
        if entry is not None
    ]
