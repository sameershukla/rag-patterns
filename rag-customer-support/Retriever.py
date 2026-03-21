"""
retriever.py — Semantic Search
Loads the pre-built FAISS index from disk and retrieves the most
relevant document chunks for a given query. Used by app.py at query time.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────

SCORE_THRESHOLD = 0.4   # Below this, no relevant doc exists — return fallback
TOP_K           = 2     # Number of chunks to retrieve per query
MODEL_NAME      = "all-MiniLM-L6-v2"  # Must match what indexer.py used


class Retriever:

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index from disk...")
        self.index = faiss.read_index("index.faiss")

        print("Loading documents metadata...")
        with open("documents.json", "r") as f:
            self.documents = json.load(f)

        print(f"Retriever ready. {self.index.ntotal} documents loaded.\n")

    def retrieve(self, query: str):
        """
        Search the index for the most relevant documents.

        Returns a list of dicts with text, score, and source.
        Returns None if no relevant document exceeds the score threshold.
        """
        # Embed the query — same model as indexing
        q_vector = self.model.encode([query], normalize_embeddings=True)
        q_vector = np.array(q_vector).astype("float32")

        # Search FAISS
        scores, indices = self.index.search(q_vector, TOP_K)

        # Check threshold — if best score is too low, nothing is relevant
        best_score = float(scores[0][0])
        if best_score < SCORE_THRESHOLD:
            return None

        # Build result list
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text":   self.documents[idx]["text"],
                "score":  round(float(scores[0][i]), 2),
                "source": self.documents[idx]["id"],
            })

        return results