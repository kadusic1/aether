from typing_extensions import TypedDict
from niche_config import Niche


class VideoState(TypedDict):
    niche: Niche
    trending_topics: list[dict]
