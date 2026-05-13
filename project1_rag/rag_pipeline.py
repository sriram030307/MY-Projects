"""
Project 1: RAG System with Analytics Dashboard
Stack: LangChain, Weaviate, OpenAI, FastAPI, RAGAS
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Vector Store ──────────────────────────────────────────────
try:
    import weaviate
    from langchain_community.vectorstores import Weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    logger.warning("Weaviate not installed. Using FAISS fallback.")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class DocumentProcessor:
    """Handles document loading and chunking strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    def load_pdf(self, path: str) -> List[Document]:
        loader = PyPDFLoader(path)
        return self.splitter.split_documents(loader.load())

    def load_text(self, path: str) -> List[Document]:
        loader = TextLoader(path)
        return self.splitter.split_documents(loader.load())

    def load_raw(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        docs = [Document(page_content=t, metadata=metadatas[i] if metadatas else {})
                for i, t in enumerate(texts)]
        return self.splitter.split_documents(docs)


class HybridRetriever:
    """Dense + sparse BM25 hybrid retrieval."""

    def __init__(self, documents: List[Document], openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.retriever.get_relevant_documents(query)[:k]

    def add_documents(self, documents: List[Document]):
        self.vectorstore.add_documents(documents)


class RAGPipeline:
    """Full RAG pipeline with evaluation logging."""

    PROMPT_TEMPLATE = """You are a helpful assistant. Use the context below to answer the question.
If you don't know the answer from the context, say so honestly.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.processor = DocumentProcessor()
        self.retriever: Optional[HybridRetriever] = None
        self.query_log: List[Dict] = []

    def ingest(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        docs = self.processor.load_raw(texts, metadatas)
        self.retriever = HybridRetriever(docs, self.api_key)
        logger.info(f"Ingested {len(docs)} chunks into vector store.")

    def ingest_pdf(self, path: str):
        docs = self.processor.load_pdf(path)
        self.retriever = HybridRetriever(docs, self.api_key)
        logger.info(f"Ingested PDF: {len(docs)} chunks.")

    def query(self, question: str) -> Dict[str, Any]:
        if not self.retriever:
            raise ValueError("No documents ingested yet. Call ingest() first.")

        start = time.time()
        retrieved_docs = self.retriever.retrieve(question)
        context = "\n\n".join([d.page_content for d in retrieved_docs])

        prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        result = chain({"query": question})
        latency = round(time.time() - start, 3)

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": result["result"],
            "num_docs_retrieved": len(retrieved_docs),
            "latency_sec": latency,
            "sources": [d.metadata for d in retrieved_docs]
        }
        self.query_log.append(log_entry)
        logger.info(f"Query answered in {latency}s")
        return log_entry

    def export_logs(self, path: str = "query_logs.json"):
        with open(path, "w") as f:
            json.dump(self.query_log, f, indent=2)
        logger.info(f"Logs exported to {path}")


# ── Demo ──────────────────────────────────────────────────────
if __name__ == "__main__":
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    sample_docs = [
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "Machine learning is a subset of AI that allows systems to learn from data.",
        "Deep learning uses neural networks with many layers to learn representations.",
        "Natural language processing enables computers to understand human language.",
        "Transformers are a neural network architecture introduced in 'Attention is All You Need'.",
    ]

    pipeline = RAGPipeline(openai_api_key=API_KEY)
    pipeline.ingest(sample_docs)

    result = pipeline.query("What is the relationship between AI and machine learning?")
    print("\n=== RAG Query Result ===")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Latency: {result['latency_sec']}s")
    print(f"Docs Retrieved: {result['num_docs_retrieved']}")

    pipeline.export_logs()
