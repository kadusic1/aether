from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable


class ContentGeneratorTemplate(ABC):
    def __init__(self, chain: Runnable):
        self.chain = chain

    def generate(self) -> str:
        headline = self.generate_headline()
        content = self.generate_content(headline)
        self.generate_short_video(headline, content)
        return content

    @abstractmethod
    def generate_headline(self) -> str:
        pass

    @abstractmethod
    def generate_content(self, headline: str) -> str:
        pass

    @abstractmethod
    def generate_short_video(self, headline: str, content: str):
        pass
