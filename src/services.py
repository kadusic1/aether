from functools import lru_cache
from langchain_community.utilities import SearxSearchWrapper


@lru_cache(maxsize=1)
def get_searx() -> SearxSearchWrapper:
    return SearxSearchWrapper(searx_host="http://localhost:8888")
