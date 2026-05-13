"""
RAG System - FastAPI Server
Endpoints: /ingest, /query, /analytics
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, json, tempfile
from datetime import datetime
from collections import Counter

from rag_pipeline import RAGPipeline

app = FastAPI(title="RAG System API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
pipeline = RAGPipeline(openai_api_key=API_KEY)


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    latency_sec: float
    num_docs_retrieved: int
    timestamp: str


@app.get("/")
def root():
    return {"message": "RAG System API is running", "docs": "/docs"}


@app.post("/ingest", summary="Ingest text documents")
def ingest(req: IngestRequest):
    try:
        pipeline.ingest(req.texts, req.metadatas)
        return {"status": "success", "chunks_ingested": len(req.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", summary="Ingest PDF file")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        pipeline.ingest_pdf(tmp_path)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, summary="Query the RAG system")
def query(req: QueryRequest):
    try:
        result = pipeline.query(req.question)
        return QueryResponse(**{k: result[k] for k in QueryResponse.__fields__})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics", summary="Get query analytics")
def analytics():
    logs = pipeline.query_log
    if not logs:
        return {"message": "No queries yet."}

    avg_latency = sum(l["latency_sec"] for l in logs) / len(logs)
    avg_docs = sum(l["num_docs_retrieved"] for l in logs) / len(logs)
    questions = [l["question"] for l in logs]

    return {
        "total_queries": len(logs),
        "avg_latency_sec": round(avg_latency, 3),
        "avg_docs_retrieved": round(avg_docs, 2),
        "recent_questions": questions[-5:],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/logs/export", summary="Export query logs as JSON")
def export_logs():
    pipeline.export_logs("query_logs.json")
    return {"status": "exported", "file": "query_logs.json"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
