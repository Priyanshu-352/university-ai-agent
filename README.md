# University AI Agent

A RAG (Retrieval-Augmented Generation) application built with Streamlit, LangChain, Google GenAI, and Qdrant vector database.

## Features

- PDF document ingestion and chunking
- Semantic search over your knowledge base
- LLM-powered answers with source citations
- Streamlit web interface

## Tech Stack

- [Streamlit](https://streamlit.io/) — Web UI
- [LangChain](https://python.langchain.com/) — RAG orchestration
- [Google GenAI](https://ai.google.dev/) — LLM & embeddings
- [Qdrant](https://qdrant.tech/) — Vector database
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF parsing

## Project Structure

```
University_AI_Agent-main/
├── rag_app.py              # Main Streamlit application
├── retrival.py             # Retrieval logic
├── langchain_chunking.py   # Document chunking utilities
├── test_qdrant_connection.py
├── requirements.txt        # Python dependencies
├── .env                    # API keys (NOT committed — see .gitignore)
├── .gitignore              # Git ignore rules
└── qdrant_storage/         # Local Qdrant data (ignored)
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd University_AI_Agent-main
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with your API keys:


### 5. Run the app

```bash
streamlit run rag_app.py
```

## License

MIT