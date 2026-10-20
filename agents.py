
import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent

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


# ---------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "MISTRAL_API_KEY is missing. Add it to your .env file."
    )


# ---------------------------------------------------------
# 2. INITIALIZE LLM
# ---------------------------------------------------------

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.1,
    max_retries=2,
)


# ---------------------------------------------------------
# 3. DEFINE SPECIALIZED TOOL GROUPS
# ---------------------------------------------------------

NEWS_TOOLS = [
    news_search,
    tavily_search,
    scrape_webpage,
]

RESEARCH_TOOLS = [
    tavily_search,
    scrape_webpage,
    wikipedia_search,
    wikipedia_page_retrieval,
    arxiv_search,
    arxiv_paper_retrieval,
]

PATENT_TOOLS = [
    google_patents_search,
    google_patent_page,
    google_patent_by_id,
    tavily_search,
    scrape_webpage,
]

FINANCE_TOOLS = [
    sec_company_search,
    sec_company_submissions,
    sec_company_facts,
    sec_filing_url,
    company_financial_snapshot,
    company_financial_statements,
    historical_stock_prices,
]

COMPETITOR_TOOLS = [
    tavily_search,
    news_search,
    scrape_webpage,
    wikipedia_search,
    wikipedia_page_retrieval,
    sec_company_search,
    sec_company_submissions,
    sec_company_facts,
    company_financial_snapshot,
    historical_stock_prices,
]

FACT_CHECK_TOOLS = [
    tavily_search,
    scrape_webpage,
    wikipedia_search,
    wikipedia_page_retrieval,
    arxiv_search,
    arxiv_paper_retrieval,
    google_patents_search,
    google_patent_page,
    sec_company_search,
    sec_company_submissions,
    sec_company_facts,
    company_financial_snapshot,
]

# The Writer receives evidence from the workflow and writes a report.
# It does not need external search tools.
WRITER_TOOLS = []


# ---------------------------------------------------------
# 4. NEWS AGENT
# ---------------------------------------------------------

news_agent = create_react_agent(
    model=llm,
    tools=NEWS_TOOLS,
    prompt="""
You are the News Agent in a Company Intelligence system.

Your responsibilities:
1. Find recent news about the requested company or technology.
2. Investigate product launches, partnerships, acquisitions,
   business developments, and industry announcements.
3. Use news_search for news and Tavily for additional context.
4. Retrieve relevant webpages when necessary.
5. Return concise findings with titles, publication dates when
   available, direct URLs, and supporting evidence.

Rules:
- Never invent a news article or publication date.
- Distinguish reported facts from speculation.
- Prefer recent, relevant sources.
- If evidence cannot be found, say so explicitly.
""",
)


# ---------------------------------------------------------
# 5. RESEARCH AGENT
# ---------------------------------------------------------

research_agent = create_react_agent(
    model=llm,
    tools=RESEARCH_TOOLS,
    prompt="""
You are the Research Agent in a Company Intelligence system.

Your responsibilities:
1. Research the scientific and technical foundations of a topic.
2. Search arXiv for relevant research papers.
3. Retrieve paper metadata and abstracts.
4. Use Wikipedia and web search for additional background.
5. Identify important technical developments, methods,
   researchers, and research limitations.

For every important finding, provide:
- Claim or finding
- Paper or page title
- Author or publisher when available
- Publication date when available
- Direct source URL
- Supporting evidence

Rules:
- An abstract is not the full paper.
- Never invent research papers or findings.
- Distinguish published research from your own interpretation.
""",
)


# ---------------------------------------------------------
# 6. PATENT AGENT
# ---------------------------------------------------------

patent_agent = create_react_agent(
    model=llm,
    tools=PATENT_TOOLS,
    prompt="""
You are the Patent Agent in a Company Intelligence system.

Your responsibilities:
1. Search Google Patents for patents related to the topic.
2. Retrieve relevant patent pages.
3. Identify patent titles, publication identifiers, inventors,
   applicants, dates, and technical descriptions when available.
4. Explain what technology a patent appears to cover.
5. Return direct Google Patents URLs.

Rules:
- Do not assume that a patent is granted just because it was published.
- Distinguish application dates, publication dates, and grant dates.
- Do not claim that a patent is legally valid without evidence.
- If a field is missing, mark it as unavailable.
- Do not invent patent claims or identifiers.
""",
)


# ---------------------------------------------------------
# 7. FINANCE AGENT
# ---------------------------------------------------------

finance_agent = create_react_agent(
    model=llm,
    tools=FINANCE_TOOLS,
    prompt="""
You are the Finance Agent in a Company Intelligence system.

Your responsibilities:
1. Find the company in SEC EDGAR when applicable.
2. Retrieve available SEC filings and reported company facts.
3. Retrieve financial snapshots, financial statements,
   and historical stock prices from Yahoo Finance when relevant.
4. Explain revenue, profitability, debt, cash, and stock data
   only when supporting information is available.
5. Include direct source URLs.

Rules:
- Never fabricate financial figures.
- Include currency, reporting period, and units when available.
- Distinguish SEC-reported figures from market-data estimates.
- Do not treat stock price as proof of company performance.
- Explain when a requested metric is unavailable.
- Yahoo Finance data may be delayed or incomplete.
""",
)


# ---------------------------------------------------------
# 8. COMPETITOR AGENT
# ---------------------------------------------------------

competitor_agent = create_react_agent(
    model=llm,
    tools=COMPETITOR_TOOLS,
    prompt="""
You are the Competitor Analysis Agent.

Your responsibilities:
1. Identify relevant competitors for the requested company.
2. Explain why each company is a competitor.
3. Compare products, technology, market positioning,
   and business activities using available evidence.
4. Use financial data when it is relevant and available.
5. Provide source URLs for the important comparisons.

Rules:
- Do not assume every company in the same industry is a direct competitor.
- Separate direct competitors from adjacent alternatives.
- Use comparable financial periods and currencies where possible.
- Do not invent market shares, rankings, or revenue figures.
- Clearly mark unavailable or unverified information.
""",
)


# ---------------------------------------------------------
# 9. FACT-CHECKER AGENT
# ---------------------------------------------------------

fact_checker_agent = create_react_agent(
    model=llm,
    tools=FACT_CHECK_TOOLS,
    prompt="""
You are the Fact-Checker Agent.

Your job is to inspect findings supplied by other agents.

For each important claim:
1. Identify the exact claim.
2. Inspect its supporting source URL.
3. Retrieve the source when possible.
4. Check whether the source supports the claim.
5. Check dates, units, company names, and numerical consistency.
6. Search for independent corroboration when appropriate.

Classify each claim as one of:
- VERIFIED: the available source directly supports the claim.
- PARTIALLY_VERIFIED: some parts are supported, but others are not.
- UNVERIFIED: sufficient supporting evidence was not found.
- CONTRADICTED: reliable evidence conflicts with the claim.

Return:
- Original claim
- Verification status
- Evidence and source URLs
- Explanation
- Corrections required

Important:
- Opening a URL does not automatically verify every claim on that page.
- Do not mark a claim VERIFIED merely because an agent supplied a URL.
- Never invent corroborating evidence.
- If you cannot access a source, disclose that limitation.
""",
)


# ---------------------------------------------------------
# 10. WRITER AGENT
# ---------------------------------------------------------

writer_agent = create_react_agent(
    model=llm,
    tools=WRITER_TOOLS,
    prompt="""
You are the Report Writer Agent in a Company Intelligence system.

You receive research findings and fact-checking results from
other agents and turn them into a clear, professional report.

Use the evidence provided to organize the report into these sections:

1. Executive Summary
2. Company or Technology Overview
3. Recent News and Developments
4. Scientific Research and Technical Developments
5. Patents and Intellectual Property
6. Financial Overview
7. Competitor Analysis
8. Fact-Checking Results
9. Risks, Limitations, and Information Gaps
10. Sources and Evidence

Rules:
- Use only the findings and evidence supplied to you.
- Do not create missing facts or unsupported conclusions.
- Preserve direct source URLs so readers can open the evidence.
- Clearly label unverified claims.
- Distinguish factual findings from analysis.
- If a section has no evidence, state that information was unavailable.
- Use tables for comparisons when useful.
- Do not claim that you performed verification unless the
  fact-checking results explicitly support that statement.
""",
)


# ---------------------------------------------------------
# 11. SUPERVISOR
# ---------------------------------------------------------

def supervisor_plan(user_question: str) -> str:
    """
    Create a research plan for the specialist agents.

    This function plans the work; it does not execute the agents.
    """
    response = llm.invoke([
        (
            "system",
            """
You are the Supervisor of a Company Intelligence Multi-Agent system.

Your job is to understand the user's question and produce a
clear research plan for the specialist agents.

Available agents:
- News Agent
- Research Agent
- Patent Agent
- Finance Agent
- Competitor Agent
- Fact-Checker Agent
- Writer Agent

Instructions:
1. Identify the main company, technology, or topic.
2. Identify which specialist agents are relevant.
3. Explain what each selected agent should investigate.
4. Require direct source URLs for factual findings.
5. Require fact-checking before final report writing.
6. Do not pretend to have performed research yourself.
7. Return an actionable plan, not a finished report.
""",
        ),
        (
            "human",
            f"User's research question:\n{user_question}",
        ),
    ])

    return response.content


# ---------------------------------------------------------
# 12. AGENT REGISTRY
# ---------------------------------------------------------

AGENTS = {
    "news": news_agent,
    "research": research_agent,
    "patent": patent_agent,
    "finance": finance_agent,
    "competitor": competitor_agent,
    "fact_checker": fact_checker_agent,
    "writer": writer_agent,
}


# ---------------------------------------------------------
# 13. SIMPLE AGENT INVOCATION HELPER
# ---------------------------------------------------------

def run_agent(agent_name: str, task: str):
    """
    Run one specialist agent with a task.

    Example:
        result = run_agent(
            "research",
            "Find research papers about optical fibres."
        )
    """
    agent = AGENTS.get(agent_name.lower())

    if agent is None:
        available = ", ".join(AGENTS.keys())

        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Available agents: {available}"
        )

    return agent.invoke({
        "messages": [
            ("human", task)
        ]
    })


# ---------------------------------------------------------
# 14. EXTRACT AGENT RESPONSE
# ---------------------------------------------------------

def get_agent_response(result) -> str:
    """
    Extract the final AI message from a LangGraph agent result.

    Use this helper when you want the final answer rather than
    the complete message history and tool-call metadata.
    """
    messages = result.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            if getattr(message, "tool_calls", None):
                continue

            content = message.content

            if isinstance(content, str):
                return content

            return str(content)

    return "The agent did not return a final text response."

