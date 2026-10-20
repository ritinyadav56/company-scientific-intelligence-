
import re

import arxiv
from langchain_core.tools import tool


@tool
def arxiv_search(query: str, max_results: int = 5) -> dict:
    """Search arXiv for scientific papers about a topic.

    Returns paper titles, authors, publication dates, abstracts,
    arXiv IDs, and direct paper URLs.
    """
    limit = max(1, min(max_results, 10))

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []

    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "published": (
                paper.published.date().isoformat()
                if paper.published
                else None
            ),
            "updated": (
                paper.updated.date().isoformat()
                if paper.updated
                else None
            ),
            "abstract": paper.summary,
            "arxiv_id": paper.get_short_id(),
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
        })

    return {
        "query": query,
        "result_count": len(results),
        "results": results,
    }


@tool
def arxiv_paper_retrieval(arxiv_id: str) -> dict:
    """Retrieve metadata and the abstract of a specific arXiv paper.

    Accepts an arXiv ID such as 1706.03762 or an arXiv paper URL.
    """
    value = arxiv_id.strip()

    # Accept either a paper ID or a URL.
    match = re.search(
        r"(?:abs/|pdf/)?(\d{4}\.\d{4,5}(?:v\d+)?)",
        value,
    )

    if not match:
        return {
            "error": (
                "Invalid arXiv ID. Example: 1706.03762 "
                "or https://arxiv.org/abs/1706.03762"
            )
        }

    paper_id = match.group(1)

    client = arxiv.Client()

    search = arxiv.Search(
        id_list=[paper_id],
        max_results=1,
    )

    paper = next(client.results(search), None)

    if paper is None:
        return {
            "arxiv_id": paper_id,
            "error": "Paper not found on arXiv.",
        }

    return {
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "published": (
            paper.published.date().isoformat()
            if paper.published
            else None
        ),
        "updated": (
            paper.updated.date().isoformat()
            if paper.updated
            else None
        ),
        "abstract": paper.summary,
        "arxiv_id": paper.get_short_id(),
        "url": paper.entry_id,
        "pdf_url": paper.pdf_url,
    }

