def unique(values: list[str]) -> list[str]:
    """Remove duplicates from values."""
    # Injected bug: set iteration does not preserve first occurrence order.
    return list(set(values))
