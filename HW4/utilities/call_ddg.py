import os
import json
import hashlib
from pathlib import Path
from langchain_community.tools import DuckDuckGoSearchResults


_searchTool = DuckDuckGoSearchResults(output_format="json")


def call_ddg(query: str) -> str:
    cached = get_cached(query)
    if cached is not None:
        return cached

    response = _searchTool.invoke(query)

    set_cached(query, response)

    return response


# CACHING


CACHE_ENABLED = os.environ.get("DDG_CACHE", "1") == "1"
CACHE_PATH = Path(os.environ.get("DDG_CACHE_PATH", ".cache/ddg.json"))

_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            try:
                _cache = json.loads(CACHE_PATH.read_text())
            except Exception:
                _cache = {}
        else:
            _cache = {}
    return _cache


def _key(messages):
    raw = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(messages):
    if not CACHE_ENABLED:
        return None
    cache = _load_cache()
    entry = cache.get(_key(messages))
    if entry is None:
        return None
    return entry[1]


def set_cached(messages, value):
    if not CACHE_ENABLED:
        return
    cache = _load_cache()
    cache[_key(messages)] = [messages, value]
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
