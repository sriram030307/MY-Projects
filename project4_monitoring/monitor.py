"""
Project 4: Automated NLP Model Monitoring System
Drift detection, performance tracking, auto-retraining triggers
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    timestamp: str
    metric: str
    baseline_mean: float
    current_mean: float
    drift_score: float
    drift_detected: bool
    threshold: float
    recommendation: str


class EmbeddingDriftDetector:
    """Detects drift in text embeddings using cosine similarity distributions."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.baseline_embeddings: Optional[np.ndarray] = None
        self._load_embedder()

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedder_available = True
            logger.info("Sentence transformer loaded.")
        except Exception as e:
            logger.warning(f"Embedder unavailable ({e}). Using random vectors.")
            self.embedder_available = False

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.embedder_available:
            return self.model.encode(texts, normalize_embeddings=True)
        # Fallback: deterministic pseudo-embeddings
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 384).astype(np.float32)

    def set_baseline(self, texts: List[str]):
        self.baseline_embeddings = self.embed(texts)
        logger.info(f"Baseline set with {len(texts)} samples.")

    def _mean_pairwise_cosine(self, A: np.ndarray, B: np.ndarray) -> float:
        """Mean cosine similarity between two sets of embeddings."""
        # Sample to avoid O(n^2) for large sets
        n = min(len(A), len(B), 100)
        A_s = A[np.random.choice(len(A), n, replace=False)]
        B_s = B[np.random.choice(len(B), n, replace=False)]
        # Dot product (already normalized)
        sims = np.dot(A_s, B_s.T).diagonal() if A_s.shape == B_s.shape else \
               np.mean(np.dot(A_s, B_s.T))
        return float(np.mean(sims))

    def detect(self, current_texts: List[str]) -> DriftReport:
        if self.baseline_embeddings is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        current_embeddings = self.embed(current_texts)
        baseline_mean = self._mean_pairwise_cosine(
            self.baseline_embeddings, self.baseline_embeddings)
        current_mean = self._mean_pairwise_cosine(
            self.baseline_embeddings, current_embeddings)
        drift_score = abs(baseline_mean - current_mean)
        drift_detected = drift_score > self.threshold

        return DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            metric="embedding_cosine_similarity",
            baseline_mean=round(baseline_mean, 4),
            current_mean=round(current_mean, 4),
            drift_score=round(drift_score, 4),
            drift_detected=drift_detected,
            threshold=self.threshold,
            recommendation="⚠️ Retrain model — distribution shift detected." if drift_detected
                           else "✅ No action needed — distribution is stable."
        )


class PerformanceMonitor:
    """Tracks prediction accuracy over time."""

    def __init__(self, degradation_threshold: float = 0.05):
        self.threshold = degradation_threshold
        self.baseline_accuracy: Optional[float] = None
        self.history: List[Dict] = []

    def set_baseline(self, accuracy: float):
        self.baseline_accuracy = accuracy
        logger.info(f"Baseline accuracy: {accuracy:.4f}")

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict:
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        degraded = (self.baseline_accuracy is not None and
                    self.baseline_accuracy - acc > self.threshold)
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "accuracy": round(acc, 4),
            "baseline_accuracy": self.baseline_accuracy,
            "degradation": round(self.baseline_accuracy - acc, 4) if self.baseline_accuracy else None,
            "performance_degraded": degraded,
            "classification_report": report
        }
        self.history.append(result)
        if degraded:
            logger.warning(f"⚠️ Performance degraded: {acc:.4f} vs baseline {self.baseline_accuracy:.4f}")
        return result


class PSICalculator:
    """Population Stability Index for feature drift detection."""

    @staticmethod
    def calculate(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> Dict:
        """PSI > 0.2 indicates significant shift."""
        baseline_pcts = np.histogram(baseline, bins=bins, density=True)[0]
        current_pcts = np.histogram(current, bins=bins, density=True)[0]
        baseline_pcts = np.clip(baseline_pcts, 1e-4, None)
        current_pcts = np.clip(current_pcts, 1e-4, None)
        psi = np.sum((current_pcts - baseline_pcts) * np.log(current_pcts / baseline_pcts))
        return {
            "psi": round(float(psi), 4),
            "severity": "🔴 Major shift" if psi > 0.2 else "🟡 Minor shift" if psi > 0.1 else "🟢 Stable"
        }


class MonitoringOrchestrator:
    """Runs all detectors and decides whether to trigger retraining."""

    def __init__(self):
        self.drift_detector = EmbeddingDriftDetector(threshold=0.15)
        self.perf_monitor = PerformanceMonitor(degradation_threshold=0.05)
        self.alerts: List[Dict] = []

    def run(self, baseline_texts: List[str], current_texts: List[str],
            y_true: List[int], y_pred: List[int],
            baseline_accuracy: float) -> Dict:

        self.drift_detector.set_baseline(baseline_texts)
        self.perf_monitor.set_baseline(baseline_accuracy)

        drift_report = self.drift_detector.detect(current_texts)
        perf_report = self.perf_monitor.evaluate(y_true, y_pred)

        retrain_needed = drift_report.drift_detected or perf_report["performance_degraded"]

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_report": asdict(drift_report),
            "performance_report": perf_report,
            "retrain_triggered": retrain_needed,
            "action": "🔄 Retrain pipeline triggered" if retrain_needed else "✅ No action needed"
        }

        if retrain_needed:
            self.alerts.append(summary)
            logger.warning(f"ALERT: Retraining triggered. Drift={drift_report.drift_detected}, "
                           f"Degraded={perf_report['performance_degraded']}")

        return summary


if __name__ == "__main__":
    orchestrator = MonitoringOrchestrator()

    baseline_texts = [
        "This product is amazing and works great",
        "Excellent quality, highly recommend",
        "Good value for money",
        "Works as expected, satisfied",
    ] * 10

    current_texts = [  # Simulated drift — different domain
        "The stock market crashed today significantly",
        "Federal reserve raises interest rates again",
        "Inflation hits 40-year high in the country",
        "Tech layoffs continue across major companies",
    ] * 10

    y_true = [1, 0, 1, 0, 1, 0, 1, 1]
    y_pred = [1, 1, 0, 0, 1, 1, 0, 1]  # Some wrong predictions

    report = orchestrator.run(
        baseline_texts=baseline_texts,
        current_texts=current_texts,
        y_true=y_true, y_pred=y_pred,
        baseline_accuracy=0.92
    )

    print("\n=== Monitoring Report ===")
    print(f"Drift Detected: {report['drift_report']['drift_detected']}")
    print(f"Drift Score: {report['drift_report']['drift_score']}")
    print(f"Recommendation: {report['drift_report']['recommendation']}")
    print(f"Current Accuracy: {report['performance_report']['accuracy']}")
    print(f"Action: {report['action']}")

    with open("monitoring_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n✅ Report saved to monitoring_report.json")
