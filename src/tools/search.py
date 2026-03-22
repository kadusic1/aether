from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from src.services import get_searx
from typing import Any, Literal
import httpx
import yt_dlp
from urllib.parse import urlencode
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


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


async def scrape_url(urls: list[str]) -> str:
    """
    Scrape and extract clean markdown content from one or more URLs.
    Scraping is done concurrently.
    Optimized for LLM consumption by removing navigation, footers, forms,
    overlays, and ignoring links/images.

    Args:
        urls: A list of web addresses to scrape (e.g., for one url ["https://site1.com"]
        or for multiple ["https://site1.com", "https://site2.com"]).

    Returns:
        A combined string of clean, formatted markdown from all successful webpages.
    """
    browser_config = BrowserConfig(
        headless=True,
        enable_stealth=True,
    )
    generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "ignore_images": True,
        }
    )
    run_config = CrawlerRunConfig(
        excluded_tags=["nav", "footer", "aside", "header"],
        remove_forms=True,
        remove_overlay_elements=True,
        markdown_generator=generator,
        page_timeout=30000,
    )

    combined_results = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            config=run_config,
            max_concurrent=3,
        )

        for i, result in enumerate(results):
            if result.success:
                # Add a header so the LLM knows which content belongs to which URL
                content = f"### Source: {urls[i]}\n{result.markdown[:5000]}\n\n"
                combined_results.append(content)
            else:
                combined_results.append(
                    f"### Source: {urls[i]}\nFailed to scrape: {result.error_message}\n\n"
                )

    # Join everything together for the LLM context window
    return "".join(combined_results)


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


def transcript_youtube_videos(video_ids: list[str]) -> str:
    """
    Fetch and format transcripts for YouTube videos.

    Args:
        video_ids: A list of YouTube video IDs to
            fetch transcripts for.

    Returns:
        A combined string of formatted transcripts,
        each prefixed with a source header. Failed
        fetches include an error message.
    """
    ytt_api = YouTubeTranscriptApi()
    formatter = TextFormatter()
    combined = []
    for video_id in video_ids:
        url = f"https://youtube.com/watch?v={video_id}"
        try:
            transcript = ytt_api.fetch(video_id)
            text = formatter.format_transcript(
                transcript=transcript,
            )
            combined.append(f"### Source: {url}\n{text[:5000]}\n\n")
        except Exception as e:
            combined.append(
                f"### Source: {url}\nFailed to fetch transcript: {e}\n\n"
            )
    return "".join(combined)
