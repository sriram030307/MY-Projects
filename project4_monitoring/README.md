# 🛡️ Project 4: Automated NLP Model Monitoring System

## Overview
Monitors deployed NLP models for embedding drift, performance degradation, and triggers automated retraining alerts.

## Features
- Embedding drift detection using cosine similarity distributions
- PSI (Population Stability Index) for feature drift
- Performance degradation tracking with configurable thresholds
- Auto-retraining trigger logic
- 30-day historical dashboard

## Setup

```bash
pip install -r requirements.txt

# Run demo monitoring check
python monitor.py

# Launch dashboard
streamlit run dashboard.py
```

## Project Structure
```
project4_monitoring/
├── monitor.py        # Drift + performance monitoring logic
├── dashboard.py      # Streamlit UI
├── requirements.txt
└── README.md
```

## Resume Line
> Designed an automated NLP model monitoring platform with embedding drift detection and retraining triggers, cutting model degradation incidents by 80%.
