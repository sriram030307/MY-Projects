"""
RAG Analytics Dashboard - Streamlit UI
Run: streamlit run dashboard.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Analytics Dashboard", layout="wide", page_icon="🧠")
st.title("🧠 RAG System — Analytics Dashboard")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("API Base URL", value=API_BASE)

# ── Ingest Documents ─────────────────────────────────────────
st.header("📄 Ingest Documents")
col1, col2 = st.columns(2)

with col1:
    raw_text = st.text_area("Paste text documents (one per line)", height=150)
    if st.button("Ingest Text"):
        texts = [t.strip() for t in raw_text.split("\n") if t.strip()]
        if texts:
            res = requests.post(f"{api_url}/ingest", json={"texts": texts})
            if res.status_code == 200:
                st.success(f"✅ Ingested {len(texts)} documents!")
            else:
                st.error(res.json().get("detail", "Error"))

with col2:
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        res = requests.post(f"{api_url}/ingest/pdf", files={"file": uploaded})
        if res.status_code == 200:
            st.success("✅ PDF ingested!")
        else:
            st.error(res.json().get("detail", "Error"))

st.divider()

# ── Query ─────────────────────────────────────────────────────
st.header("🔍 Query the RAG System")
question = st.text_input("Ask a question about your documents")
if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        res = requests.post(f"{api_url}/query", json={"question": question})
    if res.status_code == 200:
        data = res.json()
        st.success("**Answer:**")
        st.write(data["answer"])
        col1, col2, col3 = st.columns(3)
        col1.metric("⏱ Latency", f"{data['latency_sec']}s")
        col2.metric("📚 Docs Retrieved", data["num_docs_retrieved"])
        col3.metric("🕐 Timestamp", data["timestamp"][:19])
    else:
        st.error(res.json().get("detail", "Error"))

st.divider()

# ── Analytics ─────────────────────────────────────────────────
st.header("📊 Analytics")
if st.button("Refresh Analytics"):
    res = requests.get(f"{api_url}/analytics")
    if res.status_code == 200:
        data = res.json()
        if "message" in data:
            st.info(data["message"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Queries", data["total_queries"])
            col2.metric("Avg Latency", f"{data['avg_latency_sec']}s")
            col3.metric("Avg Docs Retrieved", data["avg_docs_retrieved"])

            st.subheader("Recent Questions")
            for q in data["recent_questions"]:
                st.write(f"• {q}")
    else:
        st.error("Could not fetch analytics.")
