"""
SQLite Database Manager for Microsoft Foundry Local RAG Application.
Persists document metadata, text chunks, and vector embeddings in a single self-contained SQLite file.
"""

import sqlite3
import json
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("sqlite_db")

class SQLiteDBManager:
    def __init__(self, db_path: str = "data/rag_database.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite database tables for documents and vector chunks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    filepath TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Document text chunks and vector embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    source_filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
                );
            """)
            conn.commit()
            logger.info("SQLite database initialized successfully.")

    def clear_database(self):
        """Wipe all documents and chunks from SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks;")
            cursor.execute("DELETE FROM documents;")
            conn.commit()
            logger.info("SQLite database cleared.")

    def insert_document_with_chunks(
        self,
        filename: str,
        filepath: str,
        file_size: int,
        char_count: int,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """Store document metadata and its embedded chunks into SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Remove existing document record if re-indexing
            cursor.execute("DELETE FROM documents WHERE filename = ?;", (filename,))
            
            # Insert document row
            cursor.execute("""
                INSERT INTO documents (filename, filepath, file_size, char_count)
                VALUES (?, ?, ?, ?);
            """, (filename, filepath, file_size, char_count))
            
            doc_id = cursor.lastrowid

            # Insert chunks with serialized embedding vector
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                emb_json = json.dumps(emb)
                cursor.execute("""
                    INSERT INTO document_chunks (doc_id, source_filename, chunk_index, chunk_text, embedding_json, char_count)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (doc_id, filename, idx, chunk["text"], emb_json, len(chunk["text"])))

            conn.commit()
            logger.info(f"Saved document '{filename}' with {len(chunks)} chunks to SQLite DB.")
            return doc_id

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve list of indexed documents from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.id, d.filename, d.filepath, d.file_size, d.char_count, d.uploaded_at,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.doc_id
                GROUP BY d.id;
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_chunks_with_embeddings(self) -> List[Dict[str, Any]]:
        """Fetch all chunks and deserialized embedding vectors from SQLite for similarity search."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, doc_id, source_filename, chunk_index, chunk_text, embedding_json, char_count
                FROM document_chunks;
            """)
            rows = cursor.fetchall()
            
            chunks = []
            for row in rows:
                row_dict = dict(row)
                row_dict["embedding"] = json.loads(row_dict["embedding_json"])
                del row_dict["embedding_json"]
                chunks.append(row_dict)
            return chunks

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM document_chunks;")
            chunk_count = cursor.fetchone()[0]
            
            return {
                "doc_count": doc_count,
                "chunk_count": chunk_count,
                "db_path": self.db_path
            }
