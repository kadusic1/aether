from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from src.services import get_searx
from typing import Literal


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
        None should almost never be used.
    Returns:
        A string representation of the search results including URLs.
    """
    searx = get_searx()
    results = await searx.aresults(
        query,
        num_results=min(num_results, 15),
        time_range=time_range,
    )
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
    Maximum number of URLs is 3.

    Args:
        urls: A list of web addresses to scrape (e.g., for one url ["https://site1.com"]
        or for multiple ["https://site1.com", "https://site2.com"]).

    Returns:
        A combined string of clean, formatted markdown from all successful webpages.
    """
    urls = urls[:3]
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
