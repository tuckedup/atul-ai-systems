def windows(values: list[int], size: int) -> list[list[int]]:
    """Return every contiguous window of the requested size."""
    # Injected bug: range excludes the final valid starting position.
    return [values[index:index + size] for index in range(len(values) - size)]
