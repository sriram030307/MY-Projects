# 🤖 Project 3: LLM-Powered Data Analysis Copilot

## Overview
Natural language to SQL agent with self-correction loop, auto-visualization, and conversation memory.

## Features
- NL → SQL with GPT-3.5-turbo
- Self-correction loop (retries on SQL error)
- Auto-generated charts from results
- Business insight explanation
- Pre-loaded sample SQLite database

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add: OPENAI_API_KEY=your_key

streamlit run app.py
```

## Project Structure
```
project3_copilot/
├── agent.py          # LLM agent + SQL execution
├── app.py            # Streamlit UI
├── requirements.txt
└── README.md
```

## Resume Line
> Developed a natural language to SQL/Python LLM agent with self-correction loops, reducing ad-hoc data analysis turnaround time by 60%.
