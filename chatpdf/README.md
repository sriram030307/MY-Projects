# ChatPDF – AI PDF Chat Assistant

ChatPDF is an AI-powered tool that allows users to upload PDF documents and interact with them through natural language queries. Built using Python, LangChain, and OpenAI's LLMs.

## 🚀 Features
- Upload any PDF document.
- Ask questions about the PDF content.
- Streamed conversational interface.
- Chunking and embedding for accurate results.

## 🛠 Tech Stack
- Python
- Streamlit (for UI)
- LangChain
- OpenAI API
- FAISS (Vector Store)

## 📦 Setup Instructions
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API keys in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📁 Folder Structure
```
├── app.py
├── utils.py
├── requirements.txt
├── .env
└── README.md
```

## 🧠 How it Works
- Extracts PDF content
- Chunks text into manageable pieces
- Converts chunks to vector embeddings
- Answers user queries using OpenAI with context from embeddings

## ✨ Future Scope
- Support for multiple document formats
- UI enhancements with chat history