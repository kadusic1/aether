from pydantic import BaseModel, Field


class MessageAnalysis(BaseModel):
    """
    Combined URL extraction and search routing
    from the user's message.

    Attributes:
        web_urls: List of web URLs and bare domains found
            in the user's message. Empty if none
            found.
        yt_ids: List of YouTube video IDs extracted from
            the message. Empty if none found.
        use_search: Whether the user's message
            requires a web search.
    """

    web_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs and bare domains extracted from"
            " the text. Empty list if none found."
            " DOES NOT include YouTube URLs, those are handled separately."
        ),
    )
    yt_ids: list[str] = Field(
        default_factory=list,
        description=(
            "YouTube video IDs extracted from the text. Empty list if none found."
        ),
    )
    use_search: bool = Field(
        description=(
            "True if the user's message requires a web search, False otherwise."
        ),
    )
