"""
FastAPI Server & REST API Entrypoint for Microsoft Foundry Local RAG Application.
Using SQLite database for local persistence of document chunks and embedding vectors.
"""

import os
import shutil
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_engine import RAGEngine

logger = logging.getLogger("main_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Microsoft Foundry Local - Offline RAG Knowledge Assistant",
    description="Local Q&A application built according to Microsoft Foundry Local Internship Specifications.",
    version="2.0.0"
)

# Initialize RAG Engine with SQLite storage
rag = RAGEngine(docs_dir="data/documents", db_path="data/rag_database.sqlite", chunk_size=300, chunk_overlap=60)

# Request Models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    filter_source: Optional[str] = None

# Startup Event: Load and index documents into SQLite DB
@app.on_event("startup")
def startup_event():
    logger.info("Initializing SQLite RAG Database and Vector Index...")
    rag.load_and_index_documents()

# Serves static web UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Web UI loading...</h2>")

@app.get("/api/status")
def get_status():
    """Get system, SQLite database, and Foundry Local status."""
    foundry_status = rag.client.check_connection()
    db_stats = rag.db.get_stats()
    return {
        "status": "online",
        "foundry_local": foundry_status,
        "sqlite_db": db_stats,
        "indexed_docs": db_stats["doc_count"],
        "indexed_chunks": db_stats["chunk_count"],
        "storage_engine": "SQLite Vector DB (sqlite3)"
    }

@app.get("/api/documents")
def get_documents():
    """List all indexed documents and chunks metadata from SQLite."""
    documents = rag.db.get_all_documents()
    db_stats = rag.db.get_stats()
    return {
        "doc_count": db_stats["doc_count"],
        "chunk_count": db_stats["chunk_count"],
        "documents": documents
    }

@app.post("/api/query")
def process_query(req: QueryRequest):
    """Execute local RAG query pipeline."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Soru metni boş olamaz.")
    
    db_stats = rag.db.get_stats()
    if db_stats["chunk_count"] == 0:
        raise HTTPException(status_code=400, detail="SQLite veritabanında hiç doküman yok. Lütfen önce bir doküman yükleyin.")

    result = rag.query(question=req.question, top_k=req.top_k, filter_source=req.filter_source)
    return result

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new .md, .txt, .docx, or .pdf document and persist to SQLite."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".md", ".txt", ".docx", ".pdf"]:
        raise HTTPException(status_code=400, detail="Sadece .pdf, .docx, .md ve .txt uzantılı dokümanlar kabul edilir.")
    
    save_path = os.path.join(rag.docs_dir, file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Re-index all documents into SQLite
        index_result = rag.load_and_index_documents()
        return {
            "message": f"'{file.filename}' başarıyla yüklendi ve SQLite veritabanına kaydedildi.",
            "filename": file.filename,
            "index_result": index_result
        }
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Dosya kaydedilemedi: {str(e)}")

@app.post("/api/reindex")
def reindex_all():
    """Trigger complete SQLite re-indexing."""
    result = rag.load_and_index_documents()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
