
import os
import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

REQUEST_TIMEOUT = 20
USER_AGENT = "CompanyIntelligence/1.0"


def _get_tavily_client():
    """Create a Tavily client using the API key from .env."""
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing. Add it to your .env file."
        )

    return TavilyClient(api_key=api_key)


def _validate_public_http_url(url: str):
    """Validate a URL before making an HTTP request."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("The URL must contain a hostname.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    hostname = parsed.hostname.lower()

    if hostname in ("localhost", "localhost.localdomain"):
        raise ValueError("Localhost URLs are not allowed.")

    try:
        addresses = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("Could not resolve the URL hostname.") from exc

    for address in addresses:
        ip_text = address[4][0]

        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError as exc:
            raise ValueError("Invalid IP address.") from exc

        if not ip.is_global:
            raise ValueError("Private or non-public IP addresses are not allowed.")

    return url


@tool
def tavily_search(query: str, max_results: int = 5) -> dict:
    """Search the web for company information using Tavily."""
    client = _get_tavily_client()

    response = client.search(
        query=query,
        topic="general",
        max_results=max(1, min(max_results, 10)),
    )

    return {
        "query": query,
        "results": response.get("results", []),
    }


@tool
def news_search(query: str, max_results: int = 5) -> dict:
    """Search for recent news about a company, technology, or topic."""
    client = _get_tavily_client()

    response = client.search(
        query=query,
        topic="news",
        max_results=max(1, min(max_results, 10)),
    )

    return {
        "query": query,
        "results": response.get("results", []),
    }


@tool
def scrape_webpage(url: str, max_chars: int = 12000) -> dict:
    """Extract readable text from a public webpage and return its URL."""
    _validate_public_http_url(url)

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        allow_redirects=False,
    )

    if 300 <= response.status_code < 400:
        return {
            "url": url,
            "error": "Redirect detected. Validate the destination before following it.",
        }

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" not in content_type:
        return {
            "url": url,
            "error": f"Expected an HTML page, received: {content_type}",
        }

    soup = BeautifulSoup(response.text, "lxml")

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    text_content = soup.get_text(" ", strip=True)
    limit = max(500, min(max_chars, 30000))

    return {
        "url": url,
        "title": title,
        "text": text_content[:limit],
        "truncated": len(text_content) > limit,
    }


@tool
def wikipedia_search(query: str, limit: int = 5) -> dict:
    """Search Wikipedia and return article titles and direct page URLs."""
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max(1, min(limit, 10)),
        },
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )

    response.raise_for_status()
    data = response.json()

    results = []

    for item in data.get("query", {}).get("search", []):
        title = item["title"]

        results.append({
            "title": title,
            "url": (
                "https://en.wikipedia.org/wiki/"
                + requests.utils.quote(title.replace(" ", "_"), safe="()")
            ),
            "snippet": BeautifulSoup(
                item.get("snippet", ""),
                "html.parser",
            ).get_text(" ", strip=True),
        })

    return {
        "query": query,
        "results": results,
    }


@tool
def wikipedia_page_retrieval(
    title: str,
    max_chars: int = 12000,
) -> dict:
    """Retrieve the text and direct URL of a Wikipedia article."""
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        },
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )

    response.raise_for_status()
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)

    if not page or "missing" in page:
        return {
            "title": title,
            "error": "Wikipedia article not found.",
        }

    actual_title = page.get("title", title)
    extract = page.get("extract", "")
    limit = max(500, min(max_chars, 30000))

    page_url = (
        "https://en.wikipedia.org/wiki/"
        + requests.utils.quote(
            actual_title.replace(" ", "_"),
            safe="()",
        )
    )

    return {
        "title": actual_title,
        "url": page_url,
        "text": extract[:limit],
        "truncated": len(extract) > limit,
    }


