"""
Churn Prediction Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from churn_model import generate_churn_dataset, ChurnModel, LLMExplainer, engineer_features, FEATURE_COLS

st.set_page_config(page_title="Churn Prediction Platform", layout="wide", page_icon="📉")
st.title("📉 Churn Prediction Platform with LLM Explainability")

# ── Train Model on Load ───────────────────────────────────────
@st.cache_resource
def load_model():
    df = generate_churn_dataset(1000)
    model = ChurnModel()
    metrics = model.train(df)
    return model, metrics, df

with st.spinner("Training model on sample data..."):
    model, metrics, df = load_model()

explainer = LLMExplainer()

# ── Model Performance Banner ──────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", metrics["roc_auc"])
col2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
col3.metric("Dataset Size", f"{len(df):,}")
col4.metric("Churn Rate", f"{df['churn'].mean():.1%}")

st.divider()

# ── EDA Section ───────────────────────────────────────────────
with st.expander("📊 Exploratory Data Analysis", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="tenure_months", color="churn",
                           color_discrete_map={0: "#00cc96", 1: "#ef553b"},
                           labels={"churn": "Churned"}, title="Tenure by Churn Status")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="contract_type", y="monthly_charges", color="churn",
                     color_discrete_map={0: "#00cc96", 1: "#ef553b"},
                     title="Monthly Charges by Contract Type")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        churn_by_contract = df.groupby("contract_type")["churn"].mean().reset_index()
        fig = px.bar(churn_by_contract, x="contract_type", y="churn",
                     title="Churn Rate by Contract Type", color="churn",
                     color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df.sample(200), x="tenure_months", y="monthly_charges",
                         color=df.sample(200)["churn"].map({0: "Retained", 1: "Churned"}),
                         color_discrete_map={"Retained": "#00cc96", "Churned": "#ef553b"},
                         title="Tenure vs Monthly Charges")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Single Customer Prediction ────────────────────────────────
st.header("🔍 Predict Churn for a Customer")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.slider("Tenure (months)", 1, 72, 5)
    monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 89.5)
    num_support_calls = st.slider("Support Calls", 0, 10, 7)
with col2:
    contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    payment_method = st.selectbox("Payment Method", ["Electronic Check", "Credit Card", "Bank Transfer", "Mailed Check"])
    num_products = st.slider("Number of Products", 1, 5, 1)
with col3:
    tech_support = st.checkbox("Has Tech Support", value=False)
    online_security = st.checkbox("Has Online Security", value=False)
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    avg_usage = st.slider("Avg Monthly Usage (GB)", 5.0, 50.0, 12.0)

if st.button("🚀 Predict & Explain"):
    customer = {
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": tenure * monthly_charges,
        "num_products": num_products,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "tech_support": int(tech_support),
        "online_security": int(online_security),
        "num_support_calls": num_support_calls,
        "avg_monthly_usage_gb": avg_usage,
        "region": region
    }

    with st.spinner("Analyzing customer..."):
        prediction = model.predict(customer)
        shap_values = model.get_shap_values(customer)
        explanation = explainer.explain(customer, prediction, shap_values)

    # Risk gauge
    prob = prediction["churn_probability"]
    risk_color = "#ef553b" if prob >= 0.7 else "#ffa15a" if prob >= 0.4 else "#00cc96"

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk %"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": risk_color},
                   "steps": [
                       {"range": [0, 40], "color": "#e8f5e9"},
                       {"range": [40, 70], "color": "#fff3e0"},
                       {"range": [70, 100], "color": "#ffebee"}
                   ]}
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Risk Level", prediction["risk_level"])

    with col2:
        st.subheader("💡 LLM Explanation")
        st.info(explanation)

        st.subheader("📊 Top SHAP Drivers")
        shap_df = pd.DataFrame(list(shap_values.items()), columns=["Feature", "SHAP Value"])
        shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index).head(8)
        fig = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h",
                     color="SHAP Value", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

# ── Batch Predictions ─────────────────────────────────────────
st.divider()
st.header("📋 Batch Risk Scoring")
sample_batch = df.sample(20).copy()
sample_batch_eng = sample_batch.copy()

# Quick batch prediction
try:
    from churn_model import engineer_features, FEATURE_COLS
    batch_eng = engineer_features(sample_batch)
    probs = model.model.predict_proba(batch_eng[FEATURE_COLS])[:, 1]
    sample_batch["churn_probability"] = probs.round(3)
    sample_batch["risk_level"] = pd.cut(probs, bins=[0, 0.4, 0.7, 1.0],
                                         labels=["Low", "Medium", "High"])
    display_cols = ["customer_id", "tenure_months", "monthly_charges",
                    "contract_type", "num_support_calls", "churn_probability", "risk_level"]
    st.dataframe(sample_batch[display_cols].sort_values("churn_probability", ascending=False),
                 use_container_width=True)
except Exception as e:
    st.warning(f"Batch scoring unavailable: {e}")
