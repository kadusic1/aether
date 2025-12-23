from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable


class HeadlineStrategy(ABC):
    """Abstract base class for headline generation strategies.

    Defines the interface for generating compelling headlines for video content
    using various LLM-based strategies.
    """

    @abstractmethod
    def generate(self, chain: Runnable) -> str:
        """Generate a headline for video content.

        Args:
            chain (Runnable): LLM chain to use for headline generation.

        Returns:
            str: Generated headline text.
        """
        pass


class ContentStrategy(ABC):
    """Abstract base class for content generation strategies.

    Defines the interface for generating main video content based on a headline
    using various LLM-based strategies.
    """

    @abstractmethod
    def generate(self, chain: Runnable, headline: str) -> str:
        """Generate main content for video based on a headline.

        Args:
            chain (Runnable): LLM chain to use for content generation.
            headline (str): The headline to base content generation on.

        Returns:
            str: Generated video content text.
        """
        pass


class VideoStrategy(ABC):
    """Abstract base class for video generation strategies.

    Defines the interface for generating videos based on headline and content
    using various strategies.
    """

    @abstractmethod
    def generate(self, chain: Runnable, headline: str, content: str):
        """Generate video based on headline and content.

        Args:
            chain (Runnable): LLM chain to use for video generation.
            headline (str): The video headline.
            content (str): The video content.

        Returns:
            Video output (format depends on implementation).
        """
        pass
