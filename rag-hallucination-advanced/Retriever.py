"""
retriever.py — Semantic Search + Score Logging
Loads the pre-built FAISS index and retrieves relevant documents.
Logs every retrieval score to rag_logs.jsonl for monitoring.
"""

import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────

SCORE_THRESHOLD = 0.40
TOP_K           = 2
MODEL_NAME      = "all-MiniLM-L6-v2"  # Must match indexer.py exactly
LOG_FILE        = "rag_logs.jsonl"

# ── Hallucination risk threshold ──────────────────────────────────────────────
# Scores below this are flagged as hallucination risk in the log.
# Even if above SCORE_THRESHOLD (meaning we attempt an answer),
# scores below RISK_THRESHOLD are worth monitoring.
RISK_THRESHOLD  = 0.60


class Retriever:

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index from disk...")
        self.index = faiss.read_index("index.faiss")

        print("Loading clinic documents...")
        with open("documents.json", "r") as f:
            self.documents = json.load(f)

        print(f"Retriever ready. {self.index.ntotal} documents loaded.\n")

    def retrieve(self, query: str) :
        """
        Search for the most relevant clinic documents for a given query.
        Logs every retrieval attempt to rag_logs.jsonl.

        Returns a list of result dicts (text, score, source),
        or None if no document scores above SCORE_THRESHOLD.
        """
        # Embed the query — same model as indexing
        q_vector = self.model.encode([query], normalize_embeddings=True)
        q_vector = np.array(q_vector).astype("float32")

        # Search FAISS for top-K matches
        scores, indices = self.index.search(q_vector, TOP_K)

        best_score = round(float(scores[0][0]), 3)

        # ── Log every retrieval attempt ───────────────────────────────────────
        # This builds visibility into your system over time.
        # Low scores = hallucination risk. Falling scores = stale knowledge base.
        log_entry = {
            "timestamp":          datetime.now().isoformat(),
            "query":              query,
            "best_score":         best_score,
            "answered":           best_score >= SCORE_THRESHOLD,
            "hallucination_risk": best_score < RISK_THRESHOLD,
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # ── Score threshold guard ─────────────────────────────────────────────
        # If best score is too low, no relevant doc exists.
        # Return None so app.py returns a safe fallback.
        if best_score < SCORE_THRESHOLD:
            return None

        # Build result list
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text":   self.documents[idx]["text"],
                "score":  round(float(scores[0][i]), 3),
                "source": self.documents[idx]["id"],
            })

        return results
