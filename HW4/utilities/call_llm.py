import os
import json
import hashlib
from pathlib import Path
from openai import OpenAI


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

API_MODEL = "gpt-5-mini"

from typing import TypeVar, Type, overload, Literal
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@overload
def call_llm(system_prompt: str, user_prompt: str) -> str: ...
@overload
def call_llm(system_prompt: str, user_prompt: str, response_model: Type[T]) -> T: ...


def call_llm(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T] | None = None,
) -> str | T:
    cached = get_cached(user_prompt)
    if cached is not None:
        if response_model is not None:
            cached = response_model.model_validate_json(cached)
        print(f"{"="*10} RESPONSE {"="*10} (cached)")
        print(cached)
        print(f"{"="*10} ======== {"="*10}")
        return cached

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"{"="*10} REQUEST {"="*10}")
    print(messages)
    print(f"{"="*10} ======= {"="*10}")

    if response_model is None:
        response = (
            client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
            )
            .choices[0]
            .message.content.strip()
        )

        print(f"{"="*10} RESPONSE {"="*10}")
        print(response)
        print(f"{"="*10} ======== {"="*10}")

        set_cached(user_prompt, response)
    else:
        response = client.responses.parse(
            model=API_MODEL, input=messages, text_format=response_model
        ).output_parsed

        print(f"{"="*10} RESPONSE {"="*10}")
        print(response)
        print(f"{"="*10} ======== {"="*10}")

        set_cached(user_prompt, response.model_dump_json())

    return response


# CACHING


CACHE_ENABLED = os.environ.get("LLM_CACHE", "1") == "1"
CACHE_PATH = Path(os.environ.get("LLM_CACHE_PATH", ".cache/llm.json"))

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
