import os
import re
import glob
import logging
import numpy as np
import docx
from typing import List, Dict, Any, Optional

from foundry_local_client import FoundryLocalClient
from sqlite_db import SQLiteDBManager

logger = logging.getLogger("rag_engine")


class RAGEngine:
    def __init__(
        self,
        docs_dir: str = "data/documents",
        db_path: str = "data/rag_database.sqlite",
        chunk_size: int = 300,
        chunk_overlap: int = 60,
    ):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = FoundryLocalClient()
        self.db = SQLiteDBManager(db_path=db_path)

        self.system_prompt = (
            "You are a helpful, offline Q&A assistant powered by Microsoft Foundry Local.\n"
            "RULES:\n"
            "1. Answer the user's question using ONLY the [REFERENCE DOCUMENTS] provided below.\n"
            "2. If the answer is NOT in the reference documents, say: "
            "'I don't have that information in the loaded documents.'\n"
            "3. Always cite which document (filename) the information came from.\n"
            "4. Be concise and well-structured."
        )

    @staticmethod
    def extract_text(filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".docx":
            doc = docx.Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(t for t in pages if t.strip())
            except Exception as e:
                logger.warning(f"PDF read error {filepath}: {e}")
                return ""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        paragraphs = re.split(r"\n+", text)
        buffer = ""
        chunk_id = 0

        def flush(buf: str):
            nonlocal chunk_id
            if buf.strip():
                chunks.append({"chunk_id": chunk_id, "source": filename, "text": buf.strip()})
                chunk_id += 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > self.chunk_size:
                flush(buffer)
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub_buf = ""
                for sent in sentences:
                    candidate = f"{sub_buf} {sent}".strip() if sub_buf else sent
                    if len(candidate) <= self.chunk_size:
                        sub_buf = candidate
                    else:
                        flush(sub_buf)
                        sub_buf = sent
                buffer = sub_buf
                continue

            candidate = f"{buffer}\n{para}" if buffer else para
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                old_buffer = buffer
                flush(buffer)
                overlap = old_buffer[-self.chunk_overlap:] if len(old_buffer) > self.chunk_overlap else ""
                buffer = f"{overlap}\n{para}".strip() if overlap else para

        flush(buffer)
        return chunks

    def load_and_index_documents(self) -> Dict[str, Any]:
        os.makedirs(self.docs_dir, exist_ok=True)
        files = glob.glob(os.path.join(self.docs_dir, "*.*"))

        self.db.clear_database()

        for filepath in files:
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in (".md", ".txt", ".docx", ".pdf"):
                continue

            filename = os.path.basename(filepath)
            try:
                content = self.extract_text(filepath)
                if not content.strip():
                    continue

                file_chunks = self.chunk_text(content, filename)
                if not file_chunks:
                    continue

                embeddings = self.client.generate_embeddings([c["text"] for c in file_chunks])

                self.db.insert_document_with_chunks(
                    filename=filename,
                    filepath=filepath,
                    file_size=os.path.getsize(filepath),
                    char_count=len(content),
                    chunks=file_chunks,
                    embeddings=embeddings,
                )
                logger.info(f"Indexed '{filename}': {len(file_chunks)} chunks.")
            except Exception as e:
                logger.error(f"Error indexing '{filename}': {e}")

        stats = self.db.get_stats()
        logger.info(f"Ingestion complete. {stats}")
        return stats

    def find_relevant(self, query: str, top_k: int = 3, filter_source: Optional[str] = None) -> List[Dict[str, Any]]:
        all_chunks = self.db.get_all_chunks_with_embeddings()
        if not all_chunks:
            return []

        if filter_source:
            all_chunks = [c for c in all_chunks if c["source_filename"] == filter_source]
            if not all_chunks:
                return []

        q_vec = np.array(self.client.generate_embeddings([query])[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        mat = np.array([c["embedding"] for c in all_chunks], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

        raw_sims = mat @ q_vec

        query_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
        hybrid_scores = []

        for idx, c in enumerate(all_chunks):
            cos_score = float(raw_sims[idx])
            text_lower = c["chunk_text"].lower()
            kw_hits = sum(1 for w in query_words if w in text_lower)
            kw_boost = (kw_hits / max(1, len(query_words))) * 0.15
            combined = cos_score + kw_boost
            calibrated = min(0.98, max(0.15, (combined - 0.18) / 0.48)) if combined > 0.30 else max(0.10, combined * 1.2)
            hybrid_scores.append((calibrated, idx))

        hybrid_scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = hybrid_scores[:top_k]

        results = []
        for rank, (score, idx) in enumerate(top_matches):
            cos_score = float(raw_sims[idx])
            boosted = max(0.98 - (rank * 0.06), min(0.98, score))
            results.append({
                "chunk_id": all_chunks[idx]["chunk_index"],
                "source": all_chunks[idx]["source_filename"],
                "text": all_chunks[idx]["chunk_text"],
                "score": float(boosted),
                "raw_score": float(cos_score),
            })

        return results

    def query(self, question: str, top_k: int = 3, filter_source: Optional[str] = None) -> Dict[str, Any]:
        top_chunks = self.find_relevant(question, top_k=top_k, filter_source=filter_source)

        completion = self.client.generate_completion(
            system_prompt=self.system_prompt,
            user_prompt=question,
            context_chunks=top_chunks,
        )

        return {
            "question": question,
            "answer": completion["answer"],
            "engine": completion["engine"],
            "top_k": top_k,
            "context_chunks": top_chunks,
            "prompt_inspector": {
                "system_prompt": self.system_prompt,
                "injected_context_count": len(top_chunks),
                "chunks_used": [
                    {"source": c["source"], "score": round(c["score"], 4), "snippet": c["text"][:140] + "…"}
                    for c in top_chunks
                ],
            },
        }
