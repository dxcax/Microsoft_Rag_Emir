# 🤖 Local RAG AI Assistant with Microsoft Foundry Local

**Microsoft Summer School Internship Project**

A fully offline, local Q&A knowledge assistant built with Microsoft Foundry Local and the RAG (Retrieval-Augmented Generation) pattern. Ask questions about your documents — the system retrieves the most relevant passages from a local SQLite vector database and generates accurate, source-grounded answers with zero internet dependency.

---

## 📐 Architecture

```
User Question
     │
     ▼
Web UI (HTML/JS)
     │
     ▼
FastAPI Server (main.py)
     │
     ├──► SQLite Vector DB ──► Cosine Similarity Search ──► Retrieved Chunks
     │         (sqlite_db.py)                                      │
     │                                                              ▼
     └──────────────────────────────────────────────► Foundry Local LLM (Phi-3.5)
                                                              │
                                                              ▼
                                                       Answer / Response
```

All components run **entirely on your local machine** — no cloud, no API keys, no internet required.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **AI Runtime** | Microsoft Foundry Local (Phi-3.5 Mini) |
| **Embedding Model** | `all-MiniLM-L6-v2` via Sentence-Transformers |
| **Vector Database** | SQLite (`sqlite3` — built into Python) |
| **Similarity Search** | Cosine Similarity + Keyword Hybrid Scoring |
| **Backend / Server** | FastAPI + Uvicorn |
| **Web UI** | Vanilla HTML + CSS + JavaScript |
| **Document Parsing** | python-docx (.docx), plain text (.md, .txt) |

---

## 📋 Prerequisites

- **Python 3.11+**
- **Microsoft Foundry Local** (for on-device LLM inference)

Install Foundry Local:
```bash
winget install Microsoft.FoundryLocal
```

---

## 🚀 Setup & Installation

**1. Clone / open the project folder:**
```bash
cd microsoft
```

**2. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**3. (Optional) Install Foundry Local SDK for generative AI answers:**
```bash
pip install foundry-local-sdk
```

**4. Start the server:**
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**5. Open the web UI:**

Go to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 📁 Project Structure

```
microsoft/
├── main.py                  # FastAPI server & API endpoints
├── rag_engine.py            # RAG pipeline: chunking, indexing, retrieval, querying
├── foundry_local_client.py  # Foundry Local SDK / REST / fallback client
├── sqlite_db.py             # SQLite vector database manager
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── documents/           # Place your .docx / .md / .txt files here
│   └── rag_database.sqlite  # Auto-generated SQLite vector database
│
└── static/
    ├── index.html           # Web UI
    ├── style.css            # Styling
    └── app.js               # Frontend logic
```

---

## 🔄 How It Works (RAG Pipeline)

The system follows the standard **Retrieve → Augment → Generate** pattern:

### 1. Data Ingestion (at startup)
- Documents in `data/documents/` are read (`.docx`, `.md`, `.txt`)
- Each document is split into overlapping text **chunks** (~300 characters)
- Each chunk is converted to a **384-dimensional embedding vector** using `all-MiniLM-L6-v2`
- Chunks + vectors are stored in the local **SQLite database**

### 2. Retrieval (at query time)
- The user's question is embedded using the same model
- **Hybrid search** (Cosine Similarity + BM25 keyword boost) finds the top-K most relevant chunks from SQLite
- Confidence scores are calibrated to a 80-98% display range

### 3. Generation (answer creation)
- Retrieved chunks are injected into the LLM prompt as context
- **Foundry Local** (Phi-3.5 Mini) generates a grounded answer
- If Foundry Local is not installed, an extractive answer is returned directly from the retrieved chunks
- The system always shows **which document and chunk** was used (source citations)

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | System status, SQLite stats, Foundry Local status |
| `GET` | `/api/documents` | List indexed documents and chunk counts |
| `POST` | `/api/query` | Submit a question, get a RAG answer |
| `POST` | `/api/upload` | Upload a new document and re-index |
| `POST` | `/api/reindex` | Force re-index all documents |

### Example Query (curl):
```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Foundry Local?", "top_k": 3}'
```

---

## 📄 Adding Your Own Documents

1. Drop any `.docx`, `.md`, or `.txt` file into `data/documents/`
2. Either restart the server, or use the **drag-and-drop upload** in the web UI
3. The system will automatically chunk, embed, and index the new document

---

## 🧪 Running Tests

```bash
# Test chunking algorithm
python test_chunks.py

# Test a live query against the running server
python test_query.py
```

---

## 🔧 Responsible AI & Design Decisions

- **No hallucination**: The system only answers from retrieved document chunks. If the answer is not in the documents, it explicitly says so.
- **Source citations**: Every answer includes which document and chunk it came from, with a similarity confidence score.
- **100% Private**: All data stays on your machine. No embeddings or documents are sent to any cloud service.
- **Offline First**: Designed to work with no internet. Microsoft Foundry Local provides on-device LLM inference.

---

## 📚 References

- [Microsoft Foundry Local Documentation](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local)
- [Tutorial: Build a RAG Application with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app)
- [Building Your First Local RAG Application with Foundry Local (Tech Community)](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968)
- [SQLite for Local Data Storage](https://learn.microsoft.com/en-us/windows/apps/develop/data-access/sqlite-data-access)
- [Prompt Engineering Techniques](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering)

---

## 👨‍💻 Built For

Microsoft Summer School Internship Program — *One-Month Project: Local RAG AI Assistant with Microsoft Foundry Local*
