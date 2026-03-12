# Agentic RAG Research Assistant

## 🚀 Overview
This project implements a state-of-the-art **Agentic Retrieval-Augmented Generation (RAG)** assistant. It transforms static documents into a dynamic knowledge base, orchestrated by a multi-agent system powered by **LangGraph**.

## ✨ Features
- **Intelligent Ingestion**: Uses `Unstructured` for robust parsing of PDF and TXT documents.
- **Advanced RAG**: Semantic chunking and vector storage via **Pinecone**.
- **Agentic Reasoning**: A ReAct-based Research Agent that autonomously decides when to query internal documents or search the web (**DuckDuckGo**).
- **Proactive Orchestration**: An Orchestrator Agent that manages dialogue flow and generates summary PDFs of research findings.
- **Streamlined UI**: A premium Streamlit dashboard for real-time interaction and document management.
- **Cloud-Ready**: Fully containerized with a unified startup sequence for easy deployment.

## 🏗️ Architecture
The system follows a modular architecture:
1.  **Storage Layer**: RAW files are stored in **AWS S3**. Embeddings are indexed in **Pinecone**.
2.  **Processing Pipeline**: Handles document download, Unstructured parsing, semantic chunking, and embedding generation.
3.  **Agent Network (LangGraph)**:
    - `research_agent`: Orchestrates tool usage (`vector_search_tool`, `web_search_tool`).
    - `orchestrator_agent`: Finalizes responses and handles specialized tasks like PDF generation.
4.  **API Layer (FastAPI)**: Provides asynchronous endpoints for document uploads and streaming agent responses.
5.  **Frontend (Streamlit)**: Interactive UI for querying and file uploads.

## 📂 Project Structure
```text
.
├── backend/
│   ├── api/
│   │   └── main.py          # FastAPI application & endpoints
│   ├── agents.py           # LangGraph workflow definition
│   ├── pipeline.py         # Document processing logic
│   ├── storage.py          # S3 and Pinecone integrations
│   ├── tools.py            # Agent tools (Vector, Web, PDF)
│   └── utils.py            # Logging & environment helpers
├── frontend/
│   └── app.py              # Streamlit dashboard
├── Dockerfile              # Container definition
├── start.sh                # Unified startup script
└── requirements.txt        # Python dependencies
```

## 🛠️ Setup Instructions

### Local Environment
1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure `.env`:**
    Create a `.env` file at the root (refer to Section "Environment Configuration").
4.  **Run the application:**
    - **Backend:** `python -m backend.api.main`
    - **Frontend:** `streamlit run frontend/app.py`

### 🐳 Docker & Cloud Deployment
This project is designed for high portability. Using the included `Dockerfile` and `start.sh`, you can ship the entire stack to any Linux server or Cloud VM (AWS EC2, Google Cloud Engine, Azure VM).

1.  **Build & Run:**
    ```bash
    docker build -t agentic-rag .
    docker run -p 8000:8000 -p 8501:8501 --env-file .env agentic-rag
    ```
2.  **How it works:**
    The `start.sh` script acts as a process orchestrator inside the container, launching the FastAPI backend in the background and the Streamlit frontend in the foreground. This ensures a single container can serve both the API and the UI, simplifying cloud deployments.

## 🔑 Environment Configuration
The following variables are required in your `.env` file:
| Variable | Description |
| :--- | :--- |
| `AWS_ACCESS_KEY` | AWS IAM access key with S3 permissions. |
| `AWS_SECRET_KEY` | AWS IAM secret key. |
| `AWS_REGION` | AWS region for your S3 bucket (e.g., `us-east-1`). |
| `S3_BUCKET_NAME` | The name of your S3 bucket for document storage. |
| `PINECONE_API_KEY` | Your Pinecone API key. |
| `PINECONE_INDEX_NAME` | The name of your Pinecone vector index. |
| `LLM_API_KEY` | OpenAI API key for embeddings and LLM reasoning. |

## 📡 Example API Requests

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_research_paper.pdf"
```

### Query Assistant (Streaming)
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the latest findings in the uploaded paper?", "generate_pdf": true}'
```

---

