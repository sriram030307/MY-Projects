"""
Project 5: Churn Prediction Platform with LLM Explainability Layer
EDA → Feature Engineering → XGBoost → SHAP → LLM Plain-English Explanation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Data Generation ──────────────────────────────────────────
def generate_churn_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic telecom churn dataset."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "tenure_months": rng.randint(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n).round(2),
        "total_charges": None,
        "num_products": rng.randint(1, 5, n),
        "contract_type": rng.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.5, 0.3, 0.2]),
        "payment_method": rng.choice(["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"], n),
        "tech_support": rng.choice([0, 1], n, p=[0.4, 0.6]),
        "online_security": rng.choice([0, 1], n, p=[0.45, 0.55]),
        "num_support_calls": rng.randint(0, 10, n),
        "avg_monthly_usage_gb": rng.uniform(5, 50, n).round(2),
        "region": rng.choice(["North", "South", "East", "West"], n),
    })
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"]).round(2)

    # Churn logic: higher for month-to-month, high support calls, low tenure
    churn_prob = (
        0.1 +
        0.3 * (df["contract_type"] == "Month-to-Month") +
        0.02 * df["num_support_calls"] +
        0.15 * (df["tenure_months"] < 12) -
        0.1 * df["tech_support"] -
        0.1 * df["online_security"]
    ).clip(0, 1)
    df["churn"] = (rng.random(n) < churn_prob).astype(int)
    return df


# ── Feature Engineering ───────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue_per_product"] = df["monthly_charges"] / df["num_products"]
    df["support_rate"] = df["num_support_calls"] / (df["tenure_months"] + 1)
    df["is_long_tenure"] = (df["tenure_months"] >= 24).astype(int)
    df["contract_type_enc"] = df["contract_type"].map(
        {"Month-to-Month": 0, "One Year": 1, "Two Year": 2})
    df["payment_enc"] = df["payment_method"].map(
        {"Electronic Check": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3})
    df["region_enc"] = df["region"].map({"North": 0, "South": 1, "East": 2, "West": 3})
    return df


FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges", "num_products",
    "contract_type_enc", "payment_enc", "tech_support", "online_security",
    "num_support_calls", "avg_monthly_usage_gb", "revenue_per_product",
    "support_rate", "is_long_tenure", "region_enc"
]


# ── Model Training ────────────────────────────────────────────
class ChurnModel:
    def __init__(self):
        self.model = None
        self.feature_names = FEATURE_COLS

    def train(self, df: pd.DataFrame) -> Dict:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
        from xgboost import XGBClassifier

        df = engineer_features(df)
        X = df[self.feature_names]
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
        logger.info(f"Model trained. AUC: {metrics['roc_auc']}, Accuracy: {metrics['accuracy']}")
        return metrics

    def predict(self, customer: Dict) -> Dict:
        """Predict churn probability for a single customer."""
        row = pd.DataFrame([customer])
        row = engineer_features(row)
        X = row[self.feature_names]
        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            "churn_probability": round(prob, 4),
            "churn_prediction": int(prob >= 0.5),
            "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
        }

    def get_shap_values(self, customer: Dict) -> Dict[str, float]:
        """Returns SHAP values for a single customer."""
        try:
            import shap
            row = pd.DataFrame([customer])
            row = engineer_features(row)
            X = row[self.feature_names]
            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(X)[0]
            return {feat: round(float(val), 4)
                    for feat, val in zip(self.feature_names, shap_vals)}
        except ImportError:
            # Fallback: use feature importance
            imp = self.model.feature_importances_
            return {feat: round(float(val), 4)
                    for feat, val in zip(self.feature_names, imp)}


# ── LLM Explainer ─────────────────────────────────────────────
class LLMExplainer:
    SYSTEM = """You are a customer success analyst explaining churn risk to a business manager.
Given a customer's churn probability and the top factors driving that risk (SHAP values),
write a clear, jargon-free explanation in 3-4 sentences.
Be specific. Mention the most important 2-3 risk factors by name.
End with one concrete retention recommendation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._setup()

    def _setup(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.available = True
        except Exception:
            self.available = False

    def explain(self, customer: Dict, prediction: Dict, shap_values: Dict[str, float]) -> str:
        top_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        context = f"""
Customer Profile:
- Tenure: {customer.get('tenure_months')} months
- Monthly Charges: ${customer.get('monthly_charges')}
- Contract: {customer.get('contract_type')}
- Support Calls: {customer.get('num_support_calls')}
- Tech Support: {'Yes' if customer.get('tech_support') else 'No'}

Churn Probability: {prediction['churn_probability']:.1%}
Risk Level: {prediction['risk_level']}

Top SHAP Factors (positive = increases churn risk):
{chr(10).join([f"- {feat}: {val:+.4f}" for feat, val in top_factors])}
"""
        if not self.available:
            return self._mock_explanation(customer, prediction, top_factors)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return self._mock_explanation(customer, prediction, top_factors)

    def _mock_explanation(self, customer: Dict, prediction: Dict, top_factors: List) -> str:
        risk = prediction['risk_level']
        prob = prediction['churn_probability']
        top = top_factors[0][0].replace("_", " ") if top_factors else "support calls"
        return (f"This customer has a {risk.lower()} churn risk ({prob:.1%} probability). "
                f"The strongest driver is {top}, which significantly increases their likelihood of leaving. "
                f"Additionally, their contract type and tenure suggest limited long-term commitment. "
                f"Recommendation: Offer a discounted annual contract upgrade and assign a dedicated support agent.")


if __name__ == "__main__":
    # Generate dataset
    df = generate_churn_dataset(1000)
    print(f"Dataset: {len(df)} rows, {df['churn'].mean():.1%} churn rate")

    # Train model
    model = ChurnModel()
    metrics = model.train(df)
    print(f"\nModel Metrics:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  ROC-AUC:  {metrics['roc_auc']}")

    # Predict + explain
    sample_customer = {
        "tenure_months": 5, "monthly_charges": 89.50, "total_charges": 447.50,
        "num_products": 1, "contract_type": "Month-to-Month", "payment_method": "Electronic Check",
        "tech_support": 0, "online_security": 0, "num_support_calls": 7,
        "avg_monthly_usage_gb": 12.0, "region": "North"
    }

    prediction = model.predict(sample_customer)
    shap_vals = model.get_shap_values(sample_customer)

    explainer = LLMExplainer()
    explanation = explainer.explain(sample_customer, prediction, shap_vals)

    print(f"\nPrediction:")
    print(f"  Churn Probability: {prediction['churn_probability']:.1%}")
    print(f"  Risk Level: {prediction['risk_level']}")
    print(f"\nLLM Explanation:")
    print(f"  {explanation}")
