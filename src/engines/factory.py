from langchain_core.runnables import Runnable
from src.engines.content_engine import ContentEngine
from src.strategies import (
    PsychologyHeadlineStrategy,
    PsychologyContentStrategy,
    NoVideoStrategy,
)
from src.constants import EngineType


class ContentEngineFactory:
    """Factory for creating content engines with appropriate strategies.

    This factory class provides a centralized way to instantiate ContentEngine
    objects configured with the correct combination of headline, content, and video
    strategies based on the desired engine type.
    """

    @staticmethod
    def create(engine_type: EngineType, chain: Runnable) -> ContentEngine:
        """Create a content engine with strategies appropriate for the given type.

        Args:
            engine_type: Type of content engine to create. Currently supports
                "psychology" for psychology-based content generation.
            chain: LangChain Runnable chain used for content generation.

        Returns:
            ContentEngine: A configured content engine instance with matching
                headline, content, and video strategies.

        Raises:
            ValueError: If the engine_type is not recognized or supported.
        """
        if engine_type == EngineType.PSYCHOLOGY:
            return ContentEngine(
                chain=chain,
                headline_strategy=PsychologyHeadlineStrategy(),
                content_strategy=PsychologyContentStrategy(),
                video_strategy=NoVideoStrategy(),
            )
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
