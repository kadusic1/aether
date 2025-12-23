from langchain_core.runnables import Runnable
from src.domain.interfaces import HeadlineStrategy, ContentStrategy, VideoStrategy


class ContentEngine:
    """Orchestrates content generation using configurable strategies for each phase.

    This class coordinates a multi-stage content generation pipeline that produces
    headlines, content, and video metadata for faceless channel content. It leverages
    the strategy pattern to allow flexible, pluggable implementations for each
    generation phase.

    The typical workflow is:
        1. Generate headline from the LangChain chain
        2. Generate content based on the headline
        3. Generate video metadata based on both headline and content

    Attributes:
        chain (Runnable): The LangChain runnable used for text generation.
        headline_strategy (HeadlineStrategy): Strategy responsible for headline generation.
        content_strategy (ContentStrategy): Strategy responsible for content body generation.
        video_strategy (VideoStrategy): Strategy responsible for video metadata generation.
    """

    def __init__(
        self,
        chain: Runnable,
        headline_strategy: HeadlineStrategy,
        content_strategy: ContentStrategy,
        video_strategy: VideoStrategy,
    ):
        """Initialize the content engine with required strategies and chain.

        Args:
            chain (Runnable): The LangChain runnable that will be passed to each
                strategy for text generation. This should be configured with the
                desired LLM and prompt templates.
            headline_strategy (HeadlineStrategy): Strategy implementation for
                generating engaging headlines.
            content_strategy (ContentStrategy): Strategy implementation for
                generating the main content body.
            video_strategy (VideoStrategy): Strategy implementation for generating
                video metadata and descriptions.
        """
        self.chain = chain
        self.headline_strategy = headline_strategy
        self.content_strategy = content_strategy
        self.video_strategy = video_strategy

    def generate(self) -> str:
        """Execute the complete content generation pipeline.

        Orchestrates the three-phase generation workflow:
            1. Generates a headline using the headline strategy
            2. Generates content body using the content strategy, conditioned on the headline
            3. Generates video metadata using the video strategy, informed by both headline and content

        Returns:
            str: The generated content body. Note that the headline and video
                metadata are generated as side effects but not included in the return value.
        """
        headline = self.headline_strategy.generate(self.chain)
        content = self.content_strategy.generate(self.chain, headline)
        self.video_strategy.generate(self.chain, headline, content)
        return content
