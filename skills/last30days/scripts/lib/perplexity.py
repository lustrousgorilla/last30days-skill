"""Perplexity Sonar Pro / Deep Research.

Queries Perplexity's Sonar models for AI-synthesized research with citations.
Prefers Perplexity's native API (``PERPLEXITY_API_KEY``); falls back to routing
the same models through OpenRouter (``OPENROUTER_API_KEY``) when no native key
is configured. Returns normalized items with synthesis text and individual
citation entries.

The two backends differ in three ways, handled transparently below:
  * URL              -- api.perplexity.ai vs openrouter.ai
  * model slug       -- "sonar-pro" vs "perplexity/sonar-pro"
  * citation shape   -- native top-level search_results/citations vs
                        OpenRouter's OpenAI-style message.annotations
The native path additionally sends server-side date filters, which OpenRouter's
passthrough does not expose.
"""

from __future__ import annotations

import sys
from urllib.parse import urlparse

from . import http, log


PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Native Perplexity model slugs. OpenRouter addresses the same models with a
# "perplexity/" vendor prefix.
NATIVE_SONAR_PRO = "sonar-pro"
NATIVE_DEEP_RESEARCH = "sonar-deep-research"
OPENROUTER_SONAR_PRO = "perplexity/sonar-pro"
OPENROUTER_DEEP_RESEARCH = "perplexity/sonar-deep-research"


def _log(msg: str):
    log.source_log("Perplexity", msg)


def _domain(url: str) -> str:
    return urlparse(url).netloc.strip().lower()


def _select_backend(config: dict) -> tuple[str, str] | None:
    """Pick the backend + API key, preferring the native Perplexity API.

    Returns (backend, api_key) where backend is "perplexity" or "openrouter",
    or None when neither key is configured.
    """
    px_key = config.get("PERPLEXITY_API_KEY")
    if px_key:
        return "perplexity", px_key
    or_key = config.get("OPENROUTER_API_KEY")
    if or_key:
        return "openrouter", or_key
    return None


def _to_mmddyyyy(iso_date: str) -> str:
    """Convert a YYYY-MM-DD string to Perplexity's MM/DD/YYYY filter format."""
    year, month, day = iso_date.split("-")
    return f"{month}/{day}/{year}"


def _extract_citations(backend: str, data: dict, choice: dict) -> list[dict]:
    """Normalize citations across the native and OpenRouter response shapes.

    Native Perplexity returns a top-level ``search_results`` array of
    ``{title, url, date}`` objects (preferred) plus a flat ``citations`` array
    of URL strings (fallback). OpenRouter normalizes to OpenAI-style
    ``message.annotations[].url_citation``. Returns a de-duplicated list of
    ``{"url", "title"}`` dicts in source order.
    """
    citations: list[dict] = []

    if backend == "perplexity":
        for result in data.get("search_results") or []:
            url = (result.get("url") or "").strip()
            if url:
                citations.append({"url": url, "title": (result.get("title") or "").strip()})
        if not citations:
            for url in data.get("citations") or []:
                if isinstance(url, str) and url.strip():
                    citations.append({"url": url.strip(), "title": ""})
    else:
        annotations = choice.get("message", {}).get("annotations", []) or []
        for ann in annotations:
            url_citation = ann.get("url_citation", {})
            url = (url_citation.get("url") or "").strip()
            if url:
                citations.append({"url": url, "title": (url_citation.get("title") or "").strip()})

    # Deduplicate by URL, preserving order.
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for c in citations:
        if c["url"] not in seen_urls:
            seen_urls.add(c["url"])
            unique.append(c)
    return unique


def search(
    query: str,
    date_range: tuple[str, str],
    config: dict,
    deep: bool = False,
) -> tuple[list[dict], dict]:
    """Search via Perplexity Sonar Pro or Deep Research.

    Uses the native Perplexity API when ``PERPLEXITY_API_KEY`` is set, otherwise
    routes through OpenRouter when ``OPENROUTER_API_KEY`` is set.

    Args:
        query: Search topic
        date_range: (from_date, to_date) as YYYY-MM-DD strings
        config: Must contain PERPLEXITY_API_KEY or OPENROUTER_API_KEY
        deep: Use Deep Research model (~$0.90/query) instead of Sonar Pro

    Returns:
        Tuple of (items list, artifact dict).
    """
    selection = _select_backend(config)
    if not selection:
        _log("No PERPLEXITY_API_KEY or OPENROUTER_API_KEY configured, skipping")
        return [], {}
    backend, api_key = selection

    from_date, to_date = date_range
    timeout = 120 if deep else 30

    if backend == "perplexity":
        url = PERPLEXITY_URL
        model = NATIVE_DEEP_RESEARCH if deep else NATIVE_SONAR_PRO
    else:
        url = OPENROUTER_URL
        model = OPENROUTER_DEEP_RESEARCH if deep else OPENROUTER_SONAR_PRO

    if deep:
        print("[Perplexity] Using Deep Research (~$0.90/query)", file=sys.stderr)

    prompt = (
        f"What has been happening with {query} between {from_date} and {to_date}? "
        "Include specific dates, names, numbers, and sources."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    json_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # The native API supports server-side recency filtering; OpenRouter's
    # passthrough does not surface these params, so only send them natively.
    if backend == "perplexity":
        try:
            json_data["search_after_date_filter"] = _to_mmddyyyy(from_date)
            json_data["search_before_date_filter"] = _to_mmddyyyy(to_date)
        except (ValueError, AttributeError):
            pass

    _log(f"Querying {model} via {backend} for '{query}' ({from_date} to {to_date})")

    try:
        data = http.post(url, json_data, headers=headers, timeout=timeout)
    except http.HTTPError as e:
        label = "Perplexity" if backend == "perplexity" else "OpenRouter"
        if e.status_code == 401:
            _log(f"Invalid {label} API key (401)")
        elif e.status_code == 429:
            _log(f"Rate limited by {label} (429)")
        else:
            _log(f"HTTP error: {e}")
        return [], {}
    except Exception as e:
        _log(f"Request failed: {e}")
        return [], {}

    # Parse response
    choices = data.get("choices", [])
    if not choices:
        _log("No choices in response")
        return [], {}

    synthesis = choices[0].get("message", {}).get("content", "")
    if not synthesis:
        _log("Empty synthesis content")
        return [], {}

    citations = _extract_citations(backend, data, choices[0])

    _log(f"Got synthesis ({len(synthesis)} chars) with {len(citations)} citations")

    # Build items list
    items = []

    # Primary item: the synthesis itself
    snippet = synthesis[:2000]
    items.append({
        "id": "PX1",
        "title": f"Perplexity {'Deep Research' if deep else 'Sonar Pro'}: {query}",
        "url": "",
        "source_domain": "perplexity.ai",
        "snippet": snippet,
        "date": to_date,
        "relevance": 0.9,
        "why_relevant": f"AI synthesis of recent activity for '{query}'",
        "engagement": {"citations": len(citations)},
        "metadata": {"citations": citations},
    })

    # Individual items for each citation
    for i, cit in enumerate(citations):
        items.append({
            "id": f"PX{i + 2}",
            "title": cit["title"] or _domain(cit["url"]),
            "url": cit["url"],
            "source_domain": _domain(cit["url"]),
            "snippet": "",
            "date": None,
            "relevance": 0.7,
            "why_relevant": f"Cited in Perplexity synthesis for '{query}'",
            "engagement": {"citations": 1},
            "metadata": {"citations": [cit]},
        })

    artifact = {
        "label": "perplexity",
        "backend": backend,
        "model": model,
        "deep": deep,
        "query": query,
        "synthesisLength": len(synthesis),
        "citationCount": len(citations),
    }

    return items, artifact
