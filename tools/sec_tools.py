
import os
import re

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

SEC_BASE_URL = "https://www.sec.gov"
SEC_DATA_URL = "https://data.sec.gov"

REQUEST_TIMEOUT = 30


def _sec_session():
    """Create a requests session with SEC-compliant identification."""
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()

    if not user_agent or "@" not in user_agent:
        raise ValueError(
            "Set SEC_USER_AGENT in your .env file with an application "
            "name and valid contact email address."
        )

    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    })

    return session


def _normalize_cik(cik: str) -> str:
    """Convert a CIK into its ten-digit SEC format."""
    digits = re.sub(r"\D", "", str(cik))

    if not digits:
        raise ValueError("A valid SEC CIK is required.")

    return digits.zfill(10)


def _get_json(url: str) -> dict:
    """Retrieve JSON from an SEC endpoint."""
    session = _sec_session()
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@tool
def sec_company_tickers() -> dict:
    """Retrieve the SEC company ticker mapping.

    Returns company names, stock tickers, and SEC CIK identifiers.
    """
    url = f"{SEC_BASE_URL}/files/company_tickers.json"
    data = _get_json(url)

    companies = []

    for item in data.values():
        companies.append({
            "cik": str(item.get("cik_str", "")).zfill(10),
            "name": item.get("title", ""),
            "ticker": item.get("ticker", ""),
        })

    return {
        "count": len(companies),
        "companies": companies,
        "source_url": url,
    }


@tool
def sec_company_search(query: str, limit: int = 10) -> dict:
    """Search the SEC company ticker list by company name or ticker."""
    query_clean = query.strip().lower()

    if not query_clean:
        return {
            "query": query,
            "results": [],
            "error": "Enter a company name or ticker.",
        }

    ticker_data = sec_company_tickers.invoke({})
    matches = []

    for company in ticker_data.get("companies", []):
        name = company["name"].lower()
        ticker = company["ticker"].lower()

        if query_clean in name or query_clean == ticker:
            matches.append(company)

        if len(matches) >= max(1, min(limit, 25)):
            break

    return {
        "query": query,
        "result_count": len(matches),
        "results": matches,
        "source_url": ticker_data["source_url"],
    }


@tool
def sec_company_submissions(cik: str) -> dict:
    """Retrieve a company's SEC submissions and filing history.

    The CIK can be supplied with or without leading zeros.
    """
    normalized_cik = _normalize_cik(cik)

    url = (
        f"{SEC_DATA_URL}/submissions/CIK"
        f"{normalized_cik}.json"
    )

    data = _get_json(url)

    recent = (
        data.get("filings", {})
        .get("recent", {})
    )

    filing_count = len(recent.get("form", []))
    filings = []

    for index in range(filing_count):
        accession = recent.get("accessionNumber", [""])[index]
        document = recent.get("primaryDocument", [""])[index]

        filings.append({
            "form": recent.get("form", [""])[index],
            "filing_date": recent.get("filingDate", [""])[index],
            "report_date": recent.get("reportDate", [""])[index],
            "accession_number": accession,
            "primary_document": document,
            "filing_url": (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(normalized_cik)}/"
                f"{accession.replace('-', '')}/"
                f"{document}"
                if accession and document
                else None
            ),
        })

    return {
        "company_name": data.get("name"),
        "cik": normalized_cik,
        "tickers": data.get("tickers", []),
        "exchanges": data.get("exchanges", []),
        "recent_filings": filings,
        "source_url": url,
    }


@tool
def sec_company_facts(cik: str) -> dict:
    """Retrieve XBRL company facts reported to the SEC.

    Provides reported accounting facts and their units, values,
    reporting periods, forms, and filing dates.
    """
    normalized_cik = _normalize_cik(cik)

    url = (
        f"{SEC_DATA_URL}/api/xbrl/companyfacts/"
        f"CIK{normalized_cik}.json"
    )

    data = _get_json(url)

    return {
        "company_name": data.get("entityName"),
        "cik": normalized_cik,
        "facts": data.get("facts", {}),
        "source_url": url,
    }


@tool
def sec_filing_url(
    cik: str,
    accession_number: str,
    document: str,
) -> dict:
    """Build a direct SEC EDGAR filing URL.

    Requires the company's CIK, accession number, and primary
    document filename from the SEC submissions data.
    """
    normalized_cik = _normalize_cik(cik)

    accession = accession_number.strip()
    filename = document.strip()

    if not accession or not filename:
        return {
            "error": "Accession number and document filename are required."
        }

    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        return {
            "error": "Invalid SEC accession number format."
        }

    if (
        "/" in filename
        or "\\" in filename
        or filename in (".", "..")
    ):
        return {
            "error": "Provide a document filename, not a path."
        }

    accession_without_hyphens = accession.replace("-", "")

    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(normalized_cik)}/"
        f"{accession_without_hyphens}/"
        f"{filename}"
    )

    return {
        "cik": normalized_cik,
        "accession_number": accession,
        "document": filename,
        "url": url,
        "source": "SEC EDGAR",
    }

