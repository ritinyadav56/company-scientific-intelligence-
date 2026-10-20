
import streamlit as st

from main import research_workflow


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Company Intelligence AI",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# 2. INITIALIZE TEMPORARY SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "company_context" not in st.session_state:
    st.session_state.company_context = None

if "research_count" not in st.session_state:
    st.session_state.research_count = 0


# ============================================================
# 3. CLEAR MEMORY
# ============================================================

def clear_memory():
    """Clear the current conversation and temporary context."""

    st.session_state.messages = []
    st.session_state.company_context = None
    st.session_state.research_count = 0


# ============================================================
# 4. UPDATE COMPANY CONTEXT
# ============================================================

def update_company_context(question):
    """
    Update the company context when a known company is
    explicitly mentioned in the current question.
    """

    known_companies = [
        "NVIDIA",
        "AMD",
        "Intel",
        "Qualcomm",
        "Apple",
        "Microsoft",
        "Google",
        "Alphabet",
        "Amazon",
        "Meta",
        "Tesla",
        "Samsung",
        "TSMC",
        "Broadcom",
        "Oracle",
        "IBM",
        "Reliance",
        "Infosys",
        "TCS",
    ]

    question_lower = question.lower()

    for company in known_companies:

        if company.lower() in question_lower:
            st.session_state.company_context = company
            return


# ============================================================
# 5. BUILD CONVERSATION HISTORY
# ============================================================

def build_conversation_history():
    """
    Prepare the recent conversation for main.py.

    The current question is excluded because it will be passed
    separately as user_question.
    """

    previous_messages = st.session_state.messages[-8:]

    history_parts = []

    for message in previous_messages:

        role = message["role"].upper()
        content = message["content"]

        # Avoid sending extremely long previous reports.
        if role == "ASSISTANT" and len(content) > 2000:
            content = content[:2000] + "\n[Previous report shortened]"

        history_parts.append(
            f"{role}:\n{content}"
        )

    return "\n\n".join(history_parts)


# ============================================================
# 6. PAGE HEADER
# ============================================================

st.title("🔎 Company Intelligence AI")

st.markdown(
    """
    Your AI-powered company research assistant.

    Research companies using multiple specialist agents covering:

    - 📰 News and recent developments
    - 🔬 Scientific and technical research
    - 📄 Patents and inventions
    - 💰 Financial performance
    - 🏢 Competitor analysis
    - ✅ Fact-checking and evidence
    """
)

st.caption(
    "Session memory is temporary. Ask follow-up questions "
    "without repeating the company name."
)


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Research Dashboard")

    st.metric(
        "Research Requests",
        st.session_state.research_count,
    )

    st.write(
        "**Current company:** "
        f"{st.session_state.company_context or 'Not identified'}"
    )

    st.divider()

    if st.button(
        "🧹 Clear Memory",
        use_container_width=True,
    ):

        clear_memory()
        st.rerun()

    st.divider()

    st.subheader("Example Questions")

    st.markdown(
        """
        **Company research**
        - Tell me about NVIDIA.
        - What are its latest AI technologies?

        **Financial research**
        - What is its revenue?
        - What is its profit?

        **Competitor research**
        - Who are its competitors?
        - Compare it with AMD.

        **Scientific research**
        - What research papers has it published?
        - What are its latest inventions?
        """
    )

    st.info(
        "Clear Memory removes the current chat history and "
        "company context. It does not delete reports or data "
        "saved outside this application session."
    )


# ============================================================
# 8. DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 9. CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a company intelligence question..."
)


# ============================================================
# 10. PROCESS THE QUESTION
# ============================================================

if user_question:

    user_question = user_question.strip()

    if not user_question:
        st.stop()

    # --------------------------------------------------------
    # A. HANDLE EXIT COMMANDS
    # --------------------------------------------------------

    if user_question.lower() in {
        "exit",
        "quit",
        "clear memory",
        "start new session",
    }:

        clear_memory()
        st.rerun()

    # --------------------------------------------------------
    # B. SAVE AND DISPLAY THE USER QUESTION
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # --------------------------------------------------------
    # C. UPDATE COMPANY CONTEXT
    # --------------------------------------------------------

    update_company_context(user_question)

    # --------------------------------------------------------
    # D. BUILD CONVERSATION HISTORY
    # --------------------------------------------------------

    # The current user question was just appended.
    # Exclude it from history because main.py receives it
    # separately as the user_question argument.

    conversation_history = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in st.session_state.messages[-9:-1]
    )

    # Include the current company as a context hint.
    if st.session_state.company_context:

        conversation_history += (
            "\n\nCURRENT COMPANY CONTEXT:\n"
            f"{st.session_state.company_context}"
        )

    # --------------------------------------------------------
    # E. RUN THE MULTI-AGENT WORKFLOW
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Supervisor and specialist agents are researching..."
        ):

            try:

                result = research_workflow(
                    user_question=user_question,
                    conversation_history=conversation_history,
                )

                if result is None:

                    answer = (
                        "The research workflow did not return a report. "
                        "Please check research_workflow() in main.py."
                    )

                elif isinstance(result, str):

                    answer = result.strip()

                else:

                    answer = str(result)

                if not answer:

                    answer = (
                        "The agents completed the request but "
                        "returned an empty report."
                    )

            except Exception as exc:

                answer = (
                    "An error occurred while running the research "
                    "workflow.\n\n"
                    f"**Error:** `{exc}`"
                )

        # Display the final report.
        st.markdown(answer)

    # --------------------------------------------------------
    # F. SAVE THE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.session_state.research_count += 1

