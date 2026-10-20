
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from tools.web_tools import tavily_search, scrape_webpage


@tool
def google_patents_search(query: str, max_results: int = 5) -> dict:
    """Search Google Patents for patents related to a company or technology.

    Uses Tavily to discover Google Patents pages and returns source URLs.
    """
    search_query = f"site:patents.google.com/patent {query}"

    search_result = tavily_search.invoke({
        "query": search_query,
        "max_results": max(1, min(max_results, 10)),
    })

    results = []

    for item in search_result.get("results", []):
        url = item.get("url", "")

        if "patents.google.com/patent/" not in url:
            continue

        results.append({
            "title": item.get("title", ""),
            "url": url,
            "snippet": item.get("content", ""),
            "published_date": item.get("published_date"),
            "source": "Google Patents",
        })

        if len(results) >= max(1, min(max_results, 10)):
            break

    return {
        "query": query,
        "result_count": len(results),
        "results": results,
        "note": (
            "Search results are discovered through Tavily. "
            "Verify patent details on the linked Google Patents page."
        ),
    }


@tool
def google_patent_page(url: str, max_chars: int = 15000) -> dict:
    """Retrieve readable text from a Google Patents page URL."""
    if not url.startswith("https://patents.google.com/patent/"):
        return {
            "url": url,
            "error": "Provide a valid HTTPS Google Patents page URL.",
        }

    try:
        page_result = scrape_webpage.invoke({
            "url": url,
            "max_chars": max_chars,
        })

        return {
            "url": url,
            "title": page_result.get("title", ""),
            "text": page_result.get("text", ""),
            "error": page_result.get("error"),
        }

    except Exception as exc:
        return {
            "url": url,
            "error": f"Could not retrieve patent page: {exc}",
        }


@tool
def google_patent_by_id(
    patent_id: str,
    max_chars: int = 15000,
) -> dict:
    """Retrieve a Google Patents page using its publication or patent ID.

    Example ID: US20240012345A1
    """
    cleaned_id = patent_id.strip()

    if not cleaned_id:
        return {"error": "Please provide a patent ID."}

    if "/" in cleaned_id or "\\" in cleaned_id:
        return {"error": "Provide a patent ID, not a URL or file path."}

    url = (
        "https://patents.google.com/patent/"
        + quote(cleaned_id, safe="")
    )

    return google_patent_page.invoke({
        "url": url,
        "max_chars": max_chars,
    })

