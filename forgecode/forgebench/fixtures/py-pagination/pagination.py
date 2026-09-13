def page_count(total_items: int, page_size: int) -> int:
    """Return the number of pages needed for all items."""
    # Injected bug: floor division drops a partially filled final page.
    return total_items // page_size
