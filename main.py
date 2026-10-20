
from agents import AGENTS, supervisor_plan, run_agent, get_agent_response


# ============================================================
# 1. AGENT CONFIGURATION
# ============================================================

SPECIALIST_AGENTS = {
    "news": {
        "name": "news",
        "description": "Latest company news, announcements, and updates."
    },
    "research": {
        "name": "research",
        "description": "Technology, scientific research, products, and innovation."
    },
    "patent": {
        "name": "patent",
        "description": "Patents, inventions, and intellectual property."
    },
    "finance": {
        "name": "finance",
        "description": "Revenue, profit, stock information, funding, and financial results."
    },
    "competitor": {
        "name": "competitor",
        "description": "Competitors, market position, and comparisons."
    },
}


# ============================================================
# 2. SAFE AGENT EXECUTION
# ============================================================


def execute_agent(agent_name, task):
    """
    Execute an agent using the agent name defined in agents.py.
    Return a structured result so failures are not treated as evidence.
    """

    if agent_name not in AGENTS:
        return {
            "agent": agent_name,
            "success": False,
            "content": f"Agent '{agent_name}' is not configured."
        }

    try:
        # run_agent expects a string name, not an agent object.
        result = run_agent(agent_name, task)

        # Extract the final response from the LangGraph result.
        response = get_agent_response(result)

        if not response or not str(response).strip():
            return {
                "agent": agent_name,
                "success": False,
                "content": "The agent returned an empty response."
            }

        return {
            "agent": agent_name,
            "success": True,
            "content": str(response)
        }

    except Exception as exc:
        print(f"[ERROR] {agent_name} agent failed: {exc}")

        return {
            "agent": agent_name,
            "success": False,
            "content": f"Agent execution failed: {exc}"
        }



# ============================================================
# 3. DETERMINE RELEVANT AGENTS
# ============================================================

def select_agents(question, conversation_history=""):
    """
    Select relevant agents for the current question.

    The supervisor planner is consulted first.
    Keyword rules provide a fallback and help select
    the appropriate specialist for follow-up questions.
    """

    text = f"{conversation_history}\n{question}".lower()

    selected = []

    # --------------------------------------------------------
    # Finance-related questions
    # --------------------------------------------------------

    finance_keywords = [
        "revenue", "profit", "financial", "finance",
        "funding", "valuation", "stock", "share price",
        "market cap", "market capitalization",
        "income", "earnings", "cash flow", "balance sheet",
        "quarterly results", "annual report"
    ]

    if any(keyword in text for keyword in finance_keywords):
        selected.append("finance")

    # --------------------------------------------------------
    # Competitor-related questions
    # --------------------------------------------------------

    competitor_keywords = [
        "competitor", "competitors", "competition",
        "market share", "compare", "comparison",
        "rival", "rivals", "alternative", "alternatives"
    ]

    if any(keyword in text for keyword in competitor_keywords):
        selected.append("competitor")

    # --------------------------------------------------------
    # Research and technology questions
    # --------------------------------------------------------

    research_keywords = [
        "technology", "technologies", "research",
        "innovation", "paper", "papers", "scientific",
        "architecture", "gpu", "chip", "chips",
        "product", "products", "how does", "how do",
        "invention", "future technology"
    ]

    if any(keyword in text for keyword in research_keywords):
        selected.append("research")

    # --------------------------------------------------------
    # Patent questions
    # --------------------------------------------------------

    patent_keywords = [
        "patent", "patents", "intellectual property",
        "invention filing", "patent portfolio"
    ]

    if any(keyword in text for keyword in patent_keywords):
        selected.append("patent")

    # --------------------------------------------------------
    # News-related questions
    # --------------------------------------------------------

    news_keywords = [
        "latest", "recent", "today", "news",
        "announcement", "announcements", "update",
        "updates", "this week", "breaking"
    ]

    if any(keyword in text for keyword in news_keywords):
        selected.append("news")

    # --------------------------------------------------------
    # Ask the supervisor to plan the research
    # --------------------------------------------------------

    try:
        supervisor_task = f"""
You are a research supervisor.

Select only the specialist agents needed to answer the
CURRENT QUESTION.

Available agents:
- news: latest news and company announcements
- research: technology, products, science, and innovation
- patent: patents and intellectual property
- finance: revenue, profit, stock, funding, and financial data
- competitor: competitors and market comparisons

Conversation history:
{conversation_history}

Current question:
{question}

Return only the relevant agent names from this list:
news, research, patent, finance, competitor

Do not invent agent names.
Do not answer the user's question.
"""

        plan = supervisor_plan(supervisor_task)

        # Handle common planner return types.
        if isinstance(plan, dict):
            plan_text = str(plan)
        elif isinstance(plan, list):
            plan_text = " ".join(str(item) for item in plan)
        else:
            plan_text = str(plan)

        plan_text = plan_text.lower()

        for agent_name in SPECIALIST_AGENTS:
            if agent_name in plan_text and agent_name not in selected:
                selected.append(agent_name)

    except Exception as exc:
        print(f"Supervisor planning failed: {exc}")

    # --------------------------------------------------------
    # Fallback: broad company questions
    # --------------------------------------------------------

    if not selected:
        selected = ["news", "research", "finance", "competitor"]

    # Remove duplicates while preserving order.
    selected = list(dict.fromkeys(selected))

    # Keep only configured specialist agents.
    selected = [
        name for name in selected
        if name in SPECIALIST_AGENTS and name in AGENTS
    ]

    # If the supervisor selected no usable agent, use fallback.
    if not selected:
        selected = [
            name for name in ["research", "news", "finance", "competitor"]
            if name in AGENTS
        ]

    return selected


# ============================================================
# 4. BUILD THE AGENT TASK
# ============================================================

def build_agent_task(
    user_question,
    conversation_history="",
    previous_findings=""
):
    """
    Build a task containing the current question and context.
    """

    return f"""
You are part of a company intelligence research system.

CONVERSATION HISTORY:
{conversation_history or "No previous conversation."}

CURRENT USER QUESTION:
{user_question}

FINDINGS FROM OTHER AGENTS:
{previous_findings or "No findings from other agents yet."}

Instructions:
1. Resolve references such as "it", "its", "they", and "their"
   using the conversation history.
2. Focus on the current question.
3. Do not assume that a company name or financial figure is true
   unless supported by available evidence.
4. Use dates and source URLs when the tools provide them.
5. Clearly distinguish verified information from estimates.
6. If evidence is unavailable, say so.
7. Never invent sources, URLs, figures, or research findings.
8. If another agent failed, treat its result as unavailable
   research, not as evidence.
"""


# ============================================================
# 5. RUN THE RESEARCH WORKFLOW
# ============================================================

def research_workflow(user_question, conversation_history=""):
    """
    Run research for one question.

    This function can be called by:
    - Streamlit app.py
    - The terminal interface

    Conversation history is supplied by the caller.
    This function does not save conversation history to disk.
    """

    if not user_question or not user_question.strip():
        return "Please enter a question."

    user_question = user_question.strip()

    print("\n" + "=" * 70)
    print("COMPANY INTELLIGENCE RESEARCH")
    print("=" * 70)
    print(f"Question: {user_question}")

    # --------------------------------------------------------
    # Step 1: Select relevant agents
    # --------------------------------------------------------

    selected_agents = select_agents(
        question=user_question,
        conversation_history=conversation_history
    )

    if not selected_agents:
        return (
            "No specialist agents are configured. "
            "Please check the AGENTS dictionary in agents.py."
        )

    print("\nSelected agents:", ", ".join(selected_agents))

    # --------------------------------------------------------
    # Step 2: Execute agents
    # --------------------------------------------------------

    findings = []
    successful_agents = []
    failed_agents = []

    for agent_name in selected_agents:
        print(f"\nRunning {agent_name} agent...")

        task = build_agent_task(
            user_question=user_question,
            conversation_history=conversation_history,
            previous_findings="\n\n".join(findings)
        )

        result = execute_agent(agent_name, task)

        if result["success"]:
            successful_agents.append(agent_name)

            findings.append(
                f"""
AGENT: {agent_name}

RESULT:
{result["content"]}
"""
            )

        else:
            failed_agents.append(agent_name)

            print(
                f"Warning: {agent_name} agent did not complete successfully."
            )

            # Keep failure details for debugging, but explicitly
            # mark them as unavailable evidence.
            findings.append(
                f"""
AGENT: {agent_name}

STATUS: UNAVAILABLE — NOT EVIDENCE

DETAILS:
{result["content"]}
"""
            )

    # --------------------------------------------------------
    # Step 3: Check whether any agent succeeded
    # --------------------------------------------------------

    if not successful_agents:
        return (
            "I couldn't retrieve reliable research for this question "
            "because none of the selected agents completed successfully. "
            "Please check your API keys, tool configuration, and agent logs."
        )

    all_findings = "\n\n".join(findings)

    # --------------------------------------------------------
    # Step 4: Fact-check findings
    # --------------------------------------------------------

    fact_check_prompt = f"""
You are the fact-checking agent.

USER QUESTION:
{user_question}

CONVERSATION HISTORY:
{conversation_history or "No previous conversation."}

RESEARCH FINDINGS:
{all_findings}

Your tasks:
1. Identify claims relevant to the user's question.
2. Check consistency between the supplied findings.
3. Preserve source URLs and dates supplied by the tools.
4. Identify conflicting or unsupported claims.
5. Treat all sections marked "UNAVAILABLE — NOT EVIDENCE"
   as failed research, not as factual evidence.
6. Never convert an agent error into a factual claim.
7. Do not invent sources, URLs, figures, or facts.
8. State when the available evidence is insufficient.
"""

    fact_check_result = execute_agent(
        "fact_checker",
        fact_check_prompt
    )

    if fact_check_result["success"]:
        fact_checked_findings = fact_check_result["content"]
    else:
        fact_checked_findings = (
            "Fact-checking was unavailable. Use the research findings "
            "cautiously and do not describe unverified claims as confirmed."
        )

    # --------------------------------------------------------
    # Step 5: Write the final report
    # --------------------------------------------------------

    writer_prompt = f"""
You are the final company intelligence report writer.

CURRENT USER QUESTION:
{user_question}

CONVERSATION HISTORY:
{conversation_history or "No previous conversation."}

SUCCESSFUL AGENTS:
{", ".join(successful_agents)}

FAILED AGENTS:
{", ".join(failed_agents) if failed_agents else "None"}

RAW RESEARCH FINDINGS:
{all_findings}

FACT-CHECK REVIEW:
{fact_checked_findings}

Write a clear, helpful answer to the CURRENT USER QUESTION.

Requirements:
1. Understand follow-up questions using conversation history.
2. Answer the question directly before adding extra context.
3. Use headings or bullets when useful.
4. Include source URLs when they are available in the findings.
5. Do not invent facts, figures, citations, or URLs.
6. Clearly label estimates and distinguish them from reported figures.
7. If evidence is insufficient, explain what could not be verified.
8. Never present failed-agent error messages as research findings.
9. Do not repeat the entire conversation.
10. Keep the answer focused on what the user asked.
"""

    writer_result = execute_agent("writer", writer_prompt)

    if writer_result["success"]:
        final_report = writer_result["content"]
    else:
        # Graceful fallback if the writer agent fails.
        final_report = (
            "The report-writing agent was unavailable. Here are the "
            "research findings that were retrieved:\n\n"
            + all_findings
            + "\n\nFact-check review:\n"
            + fact_checked_findings
        )

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(final_report)

    return final_report


# ============================================================
# 6. TERMINAL CHAT INTERFACE
# ============================================================

def main():
    """
    Terminal chat with temporary in-memory conversation history.

    Type:
    - clear: clear the current conversation
    - exit or quit: exit the program and discard its conversation
    """

    conversation = []

    print("\nCompany Intelligence Assistant")
    print("Ask about a company, its finances, technology, or competitors.")
    print("Type 'clear' to clear memory.")
    print("Type 'exit' or 'quit' to close the assistant.")

    while True:
        try:
            user_question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversation ended. Temporary memory discarded.")
            break

        if not user_question:
            continue

        command = user_question.lower()

        if command in {"exit", "quit"}:
            conversation.clear()
            print("Conversation ended. Temporary memory discarded.")
            break

        if command in {"clear", "clear memory", "new chat"}:
            conversation.clear()
            print("Temporary conversation memory cleared.")
            continue

        # Pass prior messages to the workflow, not the current question.
        conversation_history = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in conversation[-8:]
        )

        conversation.append({
            "role": "user",
            "content": user_question
        })

        answer = research_workflow(
            user_question=user_question,
            conversation_history=conversation_history
        )

        print("\nAssistant:", answer)

        conversation.append({
            "role": "assistant",
            "content": answer
        })


# ============================================================
# 7. SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

