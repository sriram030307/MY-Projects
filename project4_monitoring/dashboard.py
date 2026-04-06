"""
NLP Monitoring Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from monitor import MonitoringOrchestrator, PSICalculator

st.set_page_config(page_title="NLP Model Monitor", layout="wide", page_icon="🛡️")
st.title("🛡️ NLP Model Monitoring System")

# ── Simulate Historical Data ──────────────────────────────────
@st.cache_data
def generate_history():
    rows = []
    base_acc = 0.92
    for i in range(30):
        date = (datetime.utcnow() - timedelta(days=29-i)).strftime("%Y-%m-%d")
        acc = base_acc - (i * 0.002) + np.random.normal(0, 0.005)
        drift = max(0, (i - 15) * 0.01) + abs(np.random.normal(0, 0.02))
        rows.append({"date": date, "accuracy": round(acc, 4), "drift_score": round(drift, 4)})
    return rows

history = generate_history()

# ── Metrics ───────────────────────────────────────────────────
latest = history[-1]
prev = history[-2]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Accuracy", f"{latest['accuracy']:.2%}",
            f"{(latest['accuracy']-prev['accuracy']):.2%}")
col2.metric("Drift Score", f"{latest['drift_score']:.4f}",
            f"{latest['drift_score']-prev['drift_score']:+.4f}")
col3.metric("Status", "🔴 ALERT" if latest['drift_score'] > 0.15 else "🟢 Stable")
col4.metric("Days Monitored", len(history))

st.divider()

# ── Charts ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Model Accuracy Over Time")
    fig = go.Figure()
    dates = [h["date"] for h in history]
    accs = [h["accuracy"] for h in history]
    fig.add_trace(go.Scatter(x=dates, y=accs, mode="lines+markers", name="Accuracy"))
    fig.add_hline(y=0.87, line_dash="dash", line_color="red", annotation_text="Min Threshold")
    fig.update_layout(yaxis_title="Accuracy", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Embedding Drift Score Over Time")
    drifts = [h["drift_score"] for h in history]
    colors = ["red" if d > 0.15 else "green" for d in drifts]
    fig = go.Figure(go.Bar(x=dates, y=drifts, marker_color=colors))
    fig.add_hline(y=0.15, line_dash="dash", line_color="orange", annotation_text="Drift Threshold")
    fig.update_layout(yaxis_title="Drift Score", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Live Monitoring ───────────────────────────────────────────
st.subheader("🔄 Run Live Monitoring Check")

col1, col2 = st.columns(2)
with col1:
    baseline_text = st.text_area("Baseline texts (one per line)",
        "Great product, works perfectly\nHighly recommend this item\nExcellent quality and value", height=100)
with col2:
    current_text = st.text_area("Current texts (one per line)",
        "Stock market crashes today\nFed raises interest rates\nInflation hits record high", height=100)

if st.button("🚀 Run Monitoring Check"):
    baseline = [t.strip() for t in baseline_text.split("\n") if t.strip()]
    current = [t.strip() for t in current_text.split("\n") if t.strip()]

    with st.spinner("Running drift detection..."):
        orch = MonitoringOrchestrator()
        y_true = [1, 0, 1, 0][:min(4, len(current))]
        y_pred = [1, 1, 0, 0][:min(4, len(current))]
        report = orch.run(baseline, current, y_true, y_pred, 0.92)

    drift = report["drift_report"]
    perf = report["performance_report"]

    st.subheader("📋 Monitoring Report")
    col1, col2, col3 = st.columns(3)
    col1.metric("Drift Score", drift["drift_score"],
                "⚠️ Detected" if drift["drift_detected"] else "✅ OK")
    col2.metric("Current Accuracy", f"{perf['accuracy']:.2%}")
    col3.metric("Action", report["action"])

    if report["retrain_triggered"]:
        st.error("🔄 **Retraining pipeline triggered!** Model drift or degradation detected.")
    else:
        st.success("✅ Model is stable. No action needed.")

    with st.expander("Full Report JSON"):
        st.json(report)
