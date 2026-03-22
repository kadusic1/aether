from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """
    Structured output for search query generation.

    Attributes:
        query: The web search query to execute.
        youtube_query: A YouTube-specific search query
            targeting video content.
    """

    query: str = Field(
        description="The web search query to execute.",
    )
    youtube_query: str = Field(
        description=(
            "A SHORT, keyword-based YouTube search query (2-4 words maximum)."
            " Search should find videos that the audience relates to, finds"
            " entertaining, or can learn from."
        ),
    )


class SelectedResults(BaseModel):
    """
    Structured output for search result selection.

    Attributes:
        web_urls: The most interesting web URLs picked
            from search results.
        youtube_ids: YouTube video IDs selected for
            transcript extraction.
    """

    web_urls: list[str] = Field(
        min_length=3,
        description=(
            "The most interesting web URLs worth"
            " scraping for deeper insights."
            " Pick as many or as few as are"
            " genuinely relevant."
        ),
    )
    youtube_ids: list[str] = Field(
        min_length=3,
        description=(
            "YouTube video IDs selected for transcript"
            " extraction. Pick videos most likely to"
            " contain unique insights for viral"
            " short-form content."
        ),
    )
