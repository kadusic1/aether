from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable


class ContentGeneratorTemplate(ABC):
    """Abstract base class for content generation.

    This template defines the interface for generating content including headlines,
    main content, and short-form videos.

    Attributes:
        chain (Runnable): The LangChain runnable to use for text generation.
    """

    def __init__(self, chain: Runnable):
        """Initialize the content generator template.

        Args:
            chain (Runnable): The LangChain runnable for text generation.
        """
        self.chain = chain

    def generate(self) -> str:
        """Generate content by orchestrating all content generation steps.

        Returns:
            str: The generated content.
        """
        headline = self.generate_headline()
        content = self.generate_content(headline)
        self.generate_short_video(headline, content)
        return content

    @abstractmethod
    def generate_headline(self) -> str:
        """Generate a headline for the content.

        Returns:
            str: The generated headline.
        """
        pass

    @abstractmethod
    def generate_content(self, headline: str) -> str:
        """Generate the main content based on a headline.

        Args:
            headline (str): The headline to base content generation on.

        Returns:
            str: The generated content.
        """
        pass

    @abstractmethod
    def generate_short_video(self, headline: str, content: str):
        """Generate short-form video content.

        Args:
            headline (str): The headline for the video.
            content (str): The main content for the video.
        """
        pass
