"""
Multi-Agent LLM Orchestration Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import time
import json
from agents import MultiAgentOrchestrator, WorkflowState

st.set_page_config(page_title="Multi-Agent Orchestrator", layout="wide", page_icon="🤖")
st.title("🤖 Multi-Agent LLM Workflow Orchestration System")
st.caption("Researcher → Analyst → Writer → Reviewer — fully automated report generation")

# ── Agent Pipeline Visual ─────────────────────────────────────
st.subheader("🔄 Agent Pipeline")
cols = st.columns(4)
agent_info = [
    ("🔍", "Researcher", "Gathers facts & trends"),
    ("📊", "Analyst", "Identifies insights & risks"),
    ("✍️", "Writer", "Drafts structured report"),
    ("✅", "Reviewer", "Reviews & finalizes output"),
]
for col, (icon, name, desc) in zip(cols, agent_info):
    col.info(f"**{icon} {name}**\n\n{desc}")

st.divider()

# ── Input ─────────────────────────────────────────────────────
st.subheader("📝 Start a New Workflow")

sample_topics = [
    "The impact of generative AI on software development",
    "How electric vehicles are reshaping the auto industry",
    "The rise of edge computing in enterprise IT",
    "Open source LLMs vs proprietary models in 2024",
    "Data privacy regulations and their impact on AI startups",
]

selected = st.selectbox("Choose a sample topic or type your own:", ["— Type your own —"] + sample_topics)
topic_input = st.text_input("Topic:", value="" if selected == "— Type your own —" else selected)

run_btn = st.button("🚀 Run Workflow", type="primary", disabled=not topic_input.strip())

# ── Run Workflow ──────────────────────────────────────────────
if run_btn and topic_input.strip():
    orchestrator = MultiAgentOrchestrator()

    progress_bar = st.progress(0)
    status_text = st.empty()
    agent_steps = ["Researcher", "Analyst", "Writer", "Reviewer"]

    st.session_state["running"] = True

    # Live agent progress
    state = None
    for i, agent_name in enumerate(agent_steps):
        status_text.info(f"🔄 Running **{agent_name}** agent...")
        progress_bar.progress((i + 1) / len(agent_steps))

        if i == 0:
            state = orchestrator.agents[0].run(
                __import__("agents").WorkflowState(
                    task_id="live", topic=topic_input.strip()
                )
            )
        elif i == 1:
            state = orchestrator.agents[1].run(state)
        elif i == 2:
            state = orchestrator.agents[2].run(state)
        elif i == 3:
            state = orchestrator.agents[3].run(state)

        time.sleep(0.3)

    progress_bar.progress(1.0)
    status_text.success(f"✅ Workflow completed! Task ID: {state.task_id}")
    st.session_state["state"] = state
    orchestrator.save_audit_log()

# ── Display Results ───────────────────────────────────────────
if "state" in st.session_state:
    state: WorkflowState = st.session_state["state"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📄 Final Report", "🔍 Research", "📊 Analysis", "✍️ Draft", "📋 Audit Log"])

    with tab1:
        st.subheader("Final Report")
        st.markdown(state.final_report or "*No final report yet.*")
        if state.final_report:
            st.download_button(
                "⬇️ Download Report",
                data=state.final_report,
                file_name=f"report_{state.task_id}.txt",
                mime="text/plain"
            )

    with tab2:
        st.subheader("Research Notes")
        st.markdown(state.research_notes or "*Not yet generated.*")

    with tab3:
        st.subheader("Analysis")
        st.markdown(state.analysis or "*Not yet generated.*")

    with tab4:
        st.subheader("Draft Report")
        st.markdown(state.draft_report or "*Not yet generated.*")

    with tab5:
        st.subheader("Agent Audit Log")
        for msg in state.messages:
            with st.expander(f"🤖 {msg.agent} — {msg.timestamp[:19]}"):
                st.text(msg.content[:1000] + ("..." if len(msg.content) > 1000 else ""))

        # Load saved audit log if exists
        try:
            with open("audit_log.json") as f:
                audit = json.load(f)
            st.subheader("Saved Audit Log")
            st.dataframe(audit)
        except FileNotFoundError:
            pass
