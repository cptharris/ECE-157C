import os
import json
import hashlib
from pathlib import Path
from openai import OpenAI


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(system_prompt: str, user_prompt: str) -> str:
    print(f"{'='*10} QUERY LLM {'='*10}")
    print(user_prompt)

    cached = get_cached(user_prompt)
    if cached is not None:
        print(f"{'='*10} RESPONSE (cached) {'='*10}")
        print(cached)
        return cached

    response = (
        client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        .choices[0]
        .message.content.strip()
    )

    set_cached(user_prompt, response)

    print(f"{'='*10} RESPONSE (rqsted) {'='*10}")
    print(response)

    return response


# CACHING


CACHE_ENABLED = os.environ.get("LLM_CACHE", "1") == "1"
CACHE_PATH = Path(os.environ.get("LLM_CACHE_PATH", "llm_cache.json"))

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
    return cache.get(_key(messages))


def set_cached(messages, value):
    if not CACHE_ENABLED:
        return
    cache = _load_cache()
    cache[_key(messages)] = value
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
