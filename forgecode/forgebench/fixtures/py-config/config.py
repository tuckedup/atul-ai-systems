def merge_config(defaults: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    """Return a new configuration with caller overrides applied."""
    # Injected bug: defaults are applied last and overwrite caller choices.
    return {**overrides, **defaults}
