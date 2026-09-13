def enabled(value: str | bool) -> bool:
    """Interpret a configuration value as a boolean flag."""
    # Injected bug: every non-empty string, including "false", is truthy.
    return bool(value)
