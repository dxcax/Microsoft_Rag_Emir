"""
Microsoft Foundry Local Client — v3.0
Uses foundry-local-sdk v1.2.4 correct API:
  - Configuration + FoundryLocalManager.start_web_service()
  - catalog.get_model(alias).load() then .get_chat_client() / .get_embedding_client()
  - Falls back to SentenceTransformers + extractive if Foundry not available
"""

import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("foundry_client")

# Model aliases to try (in order of preference)
CHAT_MODEL_ALIASES   = ["phi-3.5-mini", "phi-4-mini", "qwen3-0.6b", "qwen2.5-0.5b"]
EMBED_MODEL_ALIASES  = ["qwen3-embedding-0.6b", "all-minilm-l6-v2"]


class FoundryLocalClient:
    def __init__(self):
        self._st_model      = None   # SentenceTransformer fallback
        self._mgr           = None   # FoundryLocalManager
        self._chat_model    = None   # loaded Model object for chat
        self._embed_model   = None   # loaded Model object for embeddings
        self._chat_alias    = None
        self._embed_alias   = None
        self.mode           = "local_fallback"
        self._init_foundry()

    # ── Foundry Local SDK Initialisation ────────────────────────

    def _init_foundry(self):
        try:
            import foundry_local_sdk as fl

            cfg = fl.Configuration(app_name="rag-app")
            self._mgr = fl.FoundryLocalManager(cfg)
            self._mgr.start_web_service()
            cat = self._mgr.catalog

            # Load chat model
            for alias in CHAT_MODEL_ALIASES:
                try:
                    m = cat.get_model(alias)
                    if m.is_cached:
                        logger.info(f"Loading chat model '{alias}' into RAM...")
                        m.load()
                        self._chat_model  = m
                        self._chat_alias  = alias
                        logger.info(f"OK Chat model '{alias}' loaded.")
                        break
                except Exception as e:
                    logger.debug(f"Chat model '{alias}' not available: {e}")

            # Load embedding model
            for alias in EMBED_MODEL_ALIASES:
                try:
                    m = cat.get_model(alias)
                    if m.is_cached:
                        logger.info(f"Loading embedding model '{alias}' into RAM...")
                        m.load()
                        self._embed_model = m
                        self._embed_alias = alias
                        logger.info(f"OK Embedding model '{alias}' loaded.")
                        break
                except Exception as e:
                    logger.debug(f"Embed model '{alias}' not available: {e}")

            if self._chat_model:
                self.mode = "foundry_sdk"
                logger.info(f"Foundry Local SDK active — chat={self._chat_alias}, embed={self._embed_alias or 'SentenceTransformers'}")
            else:
                logger.info("Foundry Local SDK started but no cached models found — using SentenceTransformers fallback.")

        except Exception as e:
            logger.info(f"Foundry Local SDK not available ({e}). Using local fallback.")
            self.mode = "local_fallback"

    def check_connection(self) -> Dict[str, Any]:
        return {
            "available": self.mode == "foundry_sdk",
            "mode": self.mode,
            "chat_model": self._chat_alias or "none",
            "embed_model": self._embed_alias or "SentenceTransformers/all-MiniLM-L6-v2",
        }

    # ── SentenceTransformers fallback ────────────────────────────

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model

    # ── Embeddings ───────────────────────────────────────────────

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Try Foundry Local embedding model
        if self._embed_model is not None:
            try:
                emb_client = self._embed_model.get_embedding_client()
                return [emb_client.embed(t) for t in texts]
            except Exception as e:
                logger.warning(f"Foundry embedding failed: {e}. Falling back to SentenceTransformers.")

        # SentenceTransformers fallback
        model = self._get_st_model()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    # ── Chat / Answer Generation ─────────────────────────────────

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.3,
    ) -> Dict[str, Any]:

        context_block = "\n\n".join([
            f"[Source: {c['source']}, Chunk #{c['chunk_id']+1}]\n{c['text']}"
            for c in context_chunks
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Answer the question using ONLY the reference documents below.\n\n"
                    f"[REFERENCE DOCUMENTS]:\n{context_block}\n\n"
                    f"[QUESTION]: {user_prompt}"
                )
            },
        ]

        # Foundry Local chat
        if self._chat_model is not None:
            try:
                chat_client = self._chat_model.get_chat_client()
                resp = chat_client.complete_chat(messages=messages)
                # Extract text content from ChatCompletion object
                if hasattr(resp, 'choices') and resp.choices:
                    answer = resp.choices[0].message.content
                elif isinstance(resp, str):
                    answer = resp
                else:
                    answer = str(resp)
                return {
                    "answer": answer,
                    "engine": f"Microsoft Foundry Local — {self._chat_alias}",
                    "sources_used": list({c["source"] for c in context_chunks}),
                }
            except Exception as e:
                logger.warning(f"Foundry chat failed: {e}")

        # Extractive fallback
        return {
            "answer": self._extractive_answer(user_prompt, context_chunks),
            "engine": "Local Extractive RAG (run: winget install Microsoft.FoundryLocal for generative AI)",
            "sources_used": list({c["source"] for c in context_chunks}),
        }

    def _extractive_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant information found in the loaded documents."

        parts = [f"### Answer for: \"{question}\"\n"]
        for i, c in enumerate(chunks, 1):
            score_pct = int(c["score"] * 100)
            parts.append(
                f"**[{i}] {c['source']}** — Chunk #{c['chunk_id']+1} | Relevance: {score_pct}%\n\n"
                f"{c['text'][:700]}\n"
            )

        parts.append(
            "\n---\n"
            "> Extractive answer from local SQLite vector database. "
            "Install Microsoft Foundry Local for generative AI answers: "
            "`winget install Microsoft.FoundryLocal`"
        )
        return "\n".join(parts)
