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

load_dotenv()

_client = None

def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set in .env")
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via Tavily.

    Args:
        query: Search query string
        max_results: Number of results to return (default 5)

    Returns:
        List of {title, url, content, score} dicts.
        Content is extracted full text, not just a snippet.
        Score is Tavily's relevance score (0.0–1.0).
    """
    client = _get_client()

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,
            search_depth="advanced",
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            })

        return results

    except Exception as e:
        print(f"[web_search] ERROR: {e}")
        return []














