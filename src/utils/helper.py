async def empty() -> str:
    """
    No-op coroutine returning an empty string.

    Used as a placeholder in asyncio.gather when one
    of the source lists is empty.
    """
    return ""
