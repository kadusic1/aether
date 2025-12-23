from langchain_core.runnables import Runnable
from src.domain.interfaces import HeadlineStrategy
from src.utils import generate_text
from prompts import system_prompt_headline


class PsychologyHeadlineStrategy(HeadlineStrategy):
    """Strategy for generating viral headlines for psychology/manipulation channels.

    This strategy produces engaging, attention-grabbing headlines optimized for
    psychology and manipulation-themed content channels using LangChain models.
    """

    def generate(self, chain: Runnable) -> str:
        """Generate a single viral headline optimized for engagement.

        Creates a compelling, attention-grabbing headline specifically designed to
        resonate with audiences interested in psychology and manipulation topics,
        maximizing click-through and engagement rates.

        Args:
            chain (Runnable): The LangChain runnable chain to use for text generation.

        Returns:
            str: A single viral headline optimized for psychology/manipulation content.
        """
        return generate_text(
            chain,
            system_prompt_headline,
            "Generate one viral headline for a psychology/manipulation channel.",
        )
