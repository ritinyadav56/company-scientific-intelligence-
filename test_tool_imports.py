
from tools.web_tools import (
    tavily_search,
    news_search,
    scrape_webpage,
    wikipedia_search,
    wikipedia_page_retrieval,
)

from tools.science_tools import (
    arxiv_search,
    arxiv_paper_retrieval,
)

from tools.patent_tools import (
    google_patents_search,
    google_patent_page,
    google_patent_by_id,
)

from tools.sec_tools import (
    sec_company_tickers,
    sec_company_search,
    sec_company_submissions,
    sec_company_facts,
    sec_filing_url,
)

from tools.finance_tools import (
    company_financial_snapshot,
    company_financial_statements,
    historical_stock_prices,
)


tools_list = [
    tavily_search,
    news_search,
    scrape_webpage,
    wikipedia_search,
    wikipedia_page_retrieval,
    arxiv_search,
    arxiv_paper_retrieval,
    google_patents_search,
    google_patent_page,
    google_patent_by_id,
    sec_company_tickers,
    sec_company_search,
    sec_company_submissions,
    sec_company_facts,
    sec_filing_url,
    company_financial_snapshot,
    company_financial_statements,
    historical_stock_prices,
]

print("\nChecking LangChain tool imports...\n")

for current_tool in tools_list:
    print(f"SUCCESS: {current_tool.name}")

print(f"\nAll {len(tools_list)} tools imported successfully!")

