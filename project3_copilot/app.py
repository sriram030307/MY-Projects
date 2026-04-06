"""
LLM Data Copilot - Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from agent import DataCopilot, create_sample_db
import os

st.set_page_config(page_title="Data Copilot", layout="wide", page_icon="🤖")
st.title("🤖 LLM-Powered Data Analysis Copilot")
st.caption("Ask questions about your data in plain English")

# ── Setup ─────────────────────────────────────────────────────
@st.cache_resource
def get_copilot():
    db = create_sample_db()
    return DataCopilot(db_path=db)

copilot = get_copilot()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("💡 Sample Questions")
sample_questions = [
    "Who are the top 5 customers by revenue?",
    "Show monthly revenue trend",
    "Which product generates the most revenue?",
    "What is the average order value by region?",
    "How many sales happened in each month?"
]
for q in sample_questions:
    if st.sidebar.button(q, key=q):
        st.session_state["question"] = q

st.sidebar.divider()
st.sidebar.header("📜 Query History")
for h in copilot.history[-5:]:
    st.sidebar.text(f"• {h['question'][:40]}...")

# ── Chat Input ────────────────────────────────────────────────
question = st.text_input("Ask anything about the data:",
                          value=st.session_state.get("question", ""),
                          placeholder="e.g. Who are the top customers by revenue?")

if st.button("Analyze 🚀") and question:
    with st.spinner("Thinking..."):
        result = copilot.query(question)

    if result["error"]:
        st.error(f"Error: {result['error']}")
    else:
        st.subheader("💡 Insight")
        st.info(result["explanation"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Results")
            st.dataframe(result["df"], use_container_width=True)
        with col2:
            st.subheader("🔍 Generated SQL")
            st.code(result["sql"], language="sql")

        # Auto-chart
        df = result["df"]
        if df is not None and len(df) > 1:
            st.subheader("📈 Auto-Visualization")
            num_cols = df.select_dtypes(include="number").columns.tolist()
            str_cols = df.select_dtypes(include="object").columns.tolist()

            if num_cols and str_cols:
                fig = px.bar(df, x=str_cols[0], y=num_cols[0],
                             color_discrete_sequence=["#636EFA"])
                st.plotly_chart(fig, use_container_width=True)
            elif len(num_cols) >= 2:
                fig = px.line(df, x=df.columns[0], y=num_cols[0])
                st.plotly_chart(fig, use_container_width=True)

    # Clear after submit
    if "question" in st.session_state:
        del st.session_state["question"]
