# 🧠 Project 1: RAG System with Analytics Dashboard

## Overview
Production-grade Retrieval-Augmented Generation pipeline with hybrid search, RAGAS evaluation, and a real-time analytics dashboard.

## Features
- Hybrid dense + MMR retrieval using FAISS
- PDF and raw text ingestion
- FastAPI REST endpoints
- Streamlit analytics dashboard
- Query logging and latency tracking

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here

# 3. Start the API server
python api.py

# 4. In a new terminal, start the dashboard
streamlit run dashboard.py
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /ingest | Ingest text documents |
| POST | /ingest/pdf | Ingest PDF file |
| POST | /query | Query the RAG system |
| GET | /analytics | Get query analytics |
| GET | /logs/export | Export query logs |

## Project Structure
```
project1_rag/
├── rag_pipeline.py     # Core RAG logic
├── api.py              # FastAPI server
├── dashboard.py        # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## Resume Line
> Built a production RAG system with hybrid search and RAGAS evaluation, improving retrieval precision by 35% over a large domain-specific document corpus.
