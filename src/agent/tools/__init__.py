"""Chemistry chatbot tools - lazy loaded to avoid circular imports."""

__all__ = ["search_compound", "search_image", "generate_speech"]


def __getattr__(name):
    """Lazy load tools."""
    if name == "search_compound":
        from .search import search_compound
        return search_compound
    elif name == "search_image":
        from .image_search import search_image
        return search_image
    elif name == "generate_speech":
        from .speech import generate_speech
        return generate_speech
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
