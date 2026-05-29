import os
import json
import hashlib
from pathlib import Path
from openai import OpenAI
from typing import TypeVar, Type, overload, Literal
from pydantic import BaseModel
import time
import random

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

API_MODEL = "gpt-5-mini"
DODEBUG = False

T = TypeVar("T", bound=BaseModel)


@overload
def call_llm(system_prompt: str, user_prompt: str, who: str = ...) -> str: ...
@overload
def call_llm(
    system_prompt: str,
    user_prompt: str,
    who: str,
    response_model: Type[T],
) -> T: ...


def call_llm(
    system_prompt: str,
    user_prompt: str,
    who: str = "general",
    response_model: Type[T] | None = None,
) -> str | T:
    cached = get_cached(who, user_prompt)
    if cached is not None:
        if response_model is not None:
            cached = response_model.model_validate_json(cached)

        if DODEBUG:
            print(f"{"="*10} RESPONSE {"="*10} (cached)")
            print(cached)
            print(f"{"="*10} ======== {"="*10}")

        # time.sleep(random.randint(2, 4))
        return cached

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if DODEBUG:
        print(f"{"="*10} REQUEST {"="*10} ({who})")
        print(user_prompt)
        print(f"{"="*10} ======= {"="*10}")

    if DODEBUG and input("continue? ") != "y":
        exit(1)

    if response_model is None:
        response = (
            client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
            )
            .choices[0]
            .message.content.strip()
        )

        set_cached(who, user_prompt, response)
    else:
        response = client.responses.parse(
            model=API_MODEL, input=messages, text_format=response_model
        ).output_parsed

        set_cached(who, user_prompt, response.model_dump_json())

    if DODEBUG:
        print("=" * 10 + " RESPONSE " + "=" * 10)
        print(response)
        print("=" * 10 + " ========" + "=" * 10)

    return response


# CACHING


CACHE_ENABLED = os.environ.get("LLM_CACHE", "1") == "1"
# default directory for cache files
CACHE_DIR = Path(os.environ.get("LLM_CACHE_PATH", ".cache/"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_file_for_who(who: str) -> Path:
    safe = "".join(c for c in str(who) if c.isalnum() or c in "-_").strip() or "general"
    return CACHE_DIR / f"llm-{safe}.json"


def _load_cache(who: str) -> dict:
    path = _cache_file_for_who(who)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_cache(who: str, cache: dict) -> None:
    path = _cache_file_for_who(who)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def _key(messages: str) -> str:
    raw = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(who: str, messages) -> str | None:
    if not CACHE_ENABLED:
        return None
    cache = _load_cache(who)
    entry = cache.get(_key(messages))
    if entry is None:
        return None
    return entry  # stored value (string or JSON-able)


def set_cached(who: str, messages, value) -> None:
    if not CACHE_ENABLED:
        return
    cache = _load_cache(who)
    cache[_key(messages)] = value
    _save_cache(who, cache)
