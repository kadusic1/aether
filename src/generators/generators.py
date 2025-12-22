from src.base import ContentGeneratorTemplate
from src.utils import generate_text
from prompts import (
    system_prompt_content,
    system_prompt_headline,
    system_prompt_review,
)


class NoVideoSimpleContentGenerator(ContentGeneratorTemplate):
    def generate_headline(self) -> str:
        return generate_text(
            self.chain,
            system_prompt_headline,
            "Generate one viral headline for a psychology/manipulation channel.",
        )

    def generate_content(self, headline: str) -> str:
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
        pass
