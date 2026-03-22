from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


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
