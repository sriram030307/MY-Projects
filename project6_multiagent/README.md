# 🤖 Project 6: Multi-Agent LLM Workflow Orchestration System

## Overview
Four specialized LLM agents (Researcher → Analyst → Writer → Reviewer) collaborate in a stateful pipeline to autonomously generate complete, polished research reports on any topic.

## Agent Pipeline
| Agent | Role |
|-------|------|
| 🔍 Researcher | Gathers key facts, trends, and data points |
| 📊 Analyst | Identifies insights, risks, and opportunities |
| ✍️ Writer | Drafts a structured business report |
| ✅ Reviewer | Reviews, scores, and finalizes the report |

## Features
- Stateful workflow with full message history
- Full audit log (agent name, timestamp, output size)
- Streamlit dashboard with live agent progress
- Downloadable final reports
- Works with or without OpenAI API key (mock mode)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add: OPENAI_API_KEY=your_key (optional — mock mode works without it)

# Run CLI demo
python agents.py

# Launch dashboard
streamlit run app.py
```

## Project Structure
```
project6_multiagent/
├── agents.py         # Agent definitions + orchestrator
├── app.py            # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Resume Line
> Architected a multi-agent LLM orchestration system with structured agent handoffs and full audit logging to automate complex research-to-report workflows end-to-end.
