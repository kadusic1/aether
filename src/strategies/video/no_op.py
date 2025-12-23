from src.domain.interfaces import VideoStrategy
from langchain_core.runnables import Runnable


class NoVideoStrategy(VideoStrategy):
    """Strategy for content without video generation.

    This strategy implements a no-op video generation approach, returning None
    to indicate that no video content should be generated.
    """

    def generate(self, chain: Runnable, headline: str, content: str) -> None:
        """No-op method that returns None instead of generating video content.

        This strategy is useful when the content pipeline should not generate
        video materials, serving as a placeholder or null object pattern implementation.

        Args:
            chain (Runnable): The LangChain runnable chain (unused).
            headline (str): The headline for the content (unused).
            content (str): The content text (unused).

        Returns:
            None: Always returns None, indicating no video generation.
        """
        return None
