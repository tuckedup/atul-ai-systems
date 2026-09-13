def mean(values: list[int]) -> float:
    """Return the arithmetic mean of a non-empty integer list."""
    # Injected bug: floor division truncates fractional means.
    return sum(values) // len(values)
