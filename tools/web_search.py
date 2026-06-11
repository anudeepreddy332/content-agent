"""
tools/web_search.py
-------------------
Tavily web search wrapper.
Returns structured results for the retrieve_node.

Why Tavily over raw Google:
- Returns clean extracted content, not just snippets
- Relevance scores built-in
- Designed for LLM pipelines — no HTML parsing needed
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient
import hashlib
import json
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential



load_dotenv()

CACHE_DIR = Path("outputs/tavily_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_DAYS = 7      # cached results valid for 7 days

_client = None

def _cache_key(query: str, max_results: int) -> str:
    return hashlib.md5(f"{query}: {max_results}".encode()).hexdigest()


def _load_cache(key: str) -> list | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    import datetime
    age_days = (datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_mtime)).days
    if age_days > CACHE_TTL_DAYS:
        path.unlink()
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # Corrupt cache entry = cache miss; live fetch overwrites it.
        # (Kept from M4 debugging — robustness, independent of the freeze.)
        return None


def _save_cache(key: str, results: list) -> None:
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(results))


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set in .env")
        _client = TavilyClient(api_key=api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)

def _call_tavily(client: TavilyClient, query: str, max_results: int) -> dict:
    """
    Inner Tavily API call with tenacity retry.

    Retries up to 3 times on any exception (network errors, timeouts, rate limits).
    reraise=True: after 3 failures the original exception propagates to web_search,
    which propagates it to retrieve_node, which logs it to error_log in state.

    Wait schedule: 2s → 4s → 8s (capped at 10s).
    """
    return client.search(
        query=query,
        max_results=max_results,
        include_raw_content=False,
        search_depth="advanced",
    )


def web_search(query: str, max_results: int = 5, force_refresh: bool = False) -> list[dict]:
    """
    Search the web via Tavily.

    Args:
        query: Search query string
        max_results: Number of results to return (default 5)
        force_refresh: If True, bypass the file cache and fetch fresh results.
                       Use when a prior pass returned suspiciously few results.

    Returns:
        List of {title, url, content, score} dicts.

    Raises:
        Exception: If all 3 Tavily retry attempts fail. Caller (retrieve_node)
                   is expected to catch this and append to error_log.
    """
    key = _cache_key(query, max_results)


    if not force_refresh:
        cached = _load_cache(key)
        if cached is not None:
            return cached

    client = _get_client()

    response = _call_tavily(client, query, max_results)

    results = []

    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        })
    _save_cache(key, results)

    return results
