from src.base import ContentGeneratorTemplate
from src.utils import generate_text
from prompts import (
    system_prompt_content,
    system_prompt_headline,
    system_prompt_review,
)


class NoVideoSimpleContentGenerator(ContentGeneratorTemplate):
    """Content generator that creates headlines and short-form items without videos.

    This generator produces viral headlines and short-form content items
    suitable for psychology/manipulation themed content channels.
    """

    def generate_headline(self) -> str:
        """Generate a viral headline for the content.

        Returns:
            str: A viral headline suitable for a psychology/manipulation channel.
        """
        return generate_text(
            self.chain,
            system_prompt_headline,
            "Generate one viral headline for a psychology/manipulation channel.",
        )

    def generate_content(self, headline: str) -> str:
        """Generate short-form content items based on the headline.

        Args:
            headline (str): The headline to base content generation on.

        Returns:
            str: The generated short-form content items.
        """
        return generate_text(
            self.chain,
            system_prompt_content,
            f"Create 6-8 viral short-form items with the headline: '{headline}'",
        )
        # A dedicated reviewer agent will be implemented later. This promotes
        # separation of concerns and cleaner code.
        # return generate_text(
        #     self.chain,
        #     system_prompt_review,
        #     f"Here is the content:\n{content}\nImprove it according to the rules.",
        # )

    def generate_short_video(self, headline: str, content: str):
        """No-op implementation since this generator doesn't produce videos.

        Args:
            headline (str): The headline (unused).
            content (str): The content (unused).
        """
        pass
