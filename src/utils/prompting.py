from src.state import VideoState


def combine_source_and_user_prompt(state: VideoState) -> str:
    """
    Combines the sources retrieved by searching and scraping with the
    original user prompt.

    Args:
        state: The workflow state

    Returns:
        String that contains sources_overview and last message extracted from
        the VideoState
    """
    parts = []
    if sources_overview := state.get("sources_overview", ""):
        parts.append(f"## RESEARCH OVERVIEW\n{sources_overview}")

    parts.append(f"## USER REQUEST\n{state['messages'][-1].content}")
    return "\n\n".join(parts)
