# prompts/__init__.py
from dataclasses import dataclass
from .psychology import psychology_trending_search_queries


@dataclass
class Niche:
    name: str
    trending_search_queries: list[str]


def niche_factory(name: str) -> Niche:
    if name == "psychology":
        return Niche(
            name="psychology",
            trending_search_queries=psychology_trending_search_queries,
        )
    else:
        raise ValueError(f"Unknown niche: {name}")


__all__ = ["Niche", "niche_factory"]
