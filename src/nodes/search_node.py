from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from niche_config.common import search_picker_prompt, search_query_prompt
from src.models import load_chat_model
from src.state import VideoState
from src.tools import search_web


class SearchQuery(BaseModel):
    """
    Structured output for search query generation.

    Attributes:
        query: The search query to execute.
        num_results: Number of results to request.
        time_range: Time range filter for recency.
    """

    query: str = Field(
        description="The search query to execute.",
    )
    num_results: int = Field(
        default=10,
        ge=5,
        le=15,
        description=("Number of search results to request (5-15)."),
    )
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default="month",
        description=(
            "Time range filter: 'day', 'week',"
            " 'month', 'year', or null for"
            " evergreen content."
        ),
    )


class SelectedResults(BaseModel):
    """
    Structured output for search result selection.

    Attributes:
        urls: The 1-3 most interesting URLs picked
            from search results.
    """

    urls: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "The 1 to 3 most interesting URLs"
            " worth scraping for deeper insights."
        ),
    )


async def search_node(state: VideoState) -> dict:
    """
    Execute a web search and pick interesting results.

    Performs two sequential LLM calls internally:
    1. Generate a niche-specific search query using
       the search_query_prompt.
    2. Evaluate raw search results and pick 1-3
       interesting URLs using the search_picker_prompt.

    The search_web tool is called directly between
    the two LLM calls (not as an LLM tool call).

    Args:
        state: The current workflow state.

    Returns:
        Dict with 'sources' key containing selected
        URLs (merged via operator.add reducer).
    """
    niche = state["niche"]
    last_message = state["messages"][-1]

    # --- Call 1: Generate search query ---
    query_prompt = search_query_prompt.format(
        persona=niche.persona,
        examples="\n".join(niche.trending_search_queries),
    )
    query_model = load_chat_model(
        temperature=0.99,
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

    # --- Execute search tool directly ---
    raw_results = await search_web(
        query_result.query,
        query_result.num_results,
        query_result.time_range,
    )

    # --- Call 2: Pick interesting URLs ---
    picker_model = load_chat_model().with_structured_output(
        SelectedResults,
    )
    picker_result = await picker_model.ainvoke(
        [
            SystemMessage(
                content=search_picker_prompt.format(
                    persona=niche.persona,
                ),
            ),
            HumanMessage(content=raw_results),
        ],
        reasoning=False,
    )

    return {"sources": picker_result.urls}
