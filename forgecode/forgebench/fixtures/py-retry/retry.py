def retry_delays(base_seconds: int, attempts: int, cap_seconds: int) -> list[int]:
    """Return capped exponential delays for retry attempts."""
    # Injected bug: exponent starts at one, so the base delay is skipped.
    return [min(cap_seconds, base_seconds * (2 ** (index + 1))) for index in range(attempts)]
