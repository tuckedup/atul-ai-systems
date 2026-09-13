from pathlib import Path


def allowed_upload(filename: str, allowed: set[str]) -> bool:
    """Return whether filename has an allowed extension."""
    # Injected bug: suffix comparison is case-sensitive.
    return Path(filename).suffix in allowed
