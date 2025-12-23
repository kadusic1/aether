from langchain_core.runnables import Runnable
from src.domain.interfaces import ContentStrategy
from src.utils import generate_text
from prompts import system_prompt_content


class PsychologyContentStrategy(ContentStrategy):
    """Strategy for generating short-form content for psychology/manipulation channels.

    This strategy creates multiple viral short-form content items (suitable for TikTok,
    Reels, Shorts, etc.) based on a provided headline using LangChain models.
    """

    def generate(self, chain: Runnable, headline: str) -> str:
        """Generate multiple short-form viral content items based on the provided headline.

        Creates 6-8 optimized short-form content snippets designed for maximum engagement
        on social media platforms, derived from the given headline and psychology/manipulation
        principles.

        Args:
            chain (Runnable): The LangChain runnable chain to use for text generation.
            headline (str): The headline or topic to base content generation on.

        Returns:
            str: Generated short-form content items, typically formatted as a list or
                series of individual content snippets.
        """
        return generate_text(
            chain,
            system_prompt_content,
            f"Create 6-8 viral short-form items with the headline: '{headline}'",
        )
