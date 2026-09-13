import time

from aisys.cache import MemoryCache


def test_memory_cache_implements_ttl_and_delete():
    cache = MemoryCache()
    cache.set("a", 1)
    assert cache.get("a") == 1
    cache.delete("a")
    assert cache.get("a") is None
    cache.set("b", 2, ttl_s=0)
    time.sleep(0.001)
    assert cache.get("b") is None
