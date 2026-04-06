# 📉 Project 5: Churn Prediction Platform with LLM Explainability

## Overview
End-to-end churn prediction with XGBoost + SHAP, plus an LLM layer that explains model decisions in plain English to business stakeholders.

## Features
- Synthetic telecom churn dataset generation
- Feature engineering (revenue per product, support rate, etc.)
- XGBoost classifier with ROC-AUC evaluation
- SHAP value computation for explainability
- LLM-generated plain-English explanations via GPT-3.5
- Interactive Streamlit dashboard with EDA, single prediction, and batch scoring

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add: OPENAI_API_KEY=your_key (optional — works without it too)

streamlit run app.py
```

## Project Structure
```
project5_churn/
├── churn_model.py    # Data generation, model training, SHAP, LLM explainer
├── app.py            # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Resume Line
> Built an end-to-end churn prediction system with SHAP-based LLM explanations, enabling non-technical stakeholders to interpret ML model decisions in plain English.
