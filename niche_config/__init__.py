from dataclasses import dataclass

from .psychology import (
    psychology_niche_persona,
    psychology_trending_search_queries,
)
from .common import base_persona


@dataclass
class Niche:
    """
    Configuration data for a specific content niche.

    Each niche carries its own persona string which is
    injected into the common prompts to provide
    niche-tailored instructions.

    Attributes:
        name: The identifier name of the niche.
        persona: The full persona instructions combining
            base AI role + niche specific role.
        trending_search_queries: Example queries used
            as inspiration for research.
    """

    name: str
    persona: str
    trending_search_queries: list[str]


def niche_factory(name: str) -> Niche:
    """
    Create and return a Niche configuration instance by name.
    Args:
        name: The name of the niche to load.
    Returns:
        The populated Niche dataclass.
    Raises:
        ValueError: If the requested niche name is not recognized.
    """
    if name == "psychology":
        full_persona = f"{base_persona}\n{psychology_niche_persona}"
        return Niche(
            name="psychology",
            persona=full_persona,
            trending_search_queries=psychology_trending_search_queries,
        )
    else:
        raise ValueError(f"Unknown niche: {name}")


__all__ = ["Niche", "niche_factory"]
