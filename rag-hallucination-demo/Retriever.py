"""
retriever.py — Semantic Search
Loads the pre-built FAISS index from disk and retrieves the most
relevant clinic documents for a given query.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────

SCORE_THRESHOLD = 0.40   # Minimum similarity score to consider a doc relevant
TOP_K           = 2      # Number of documents to retrieve per query
MODEL_NAME      = "all-MiniLM-L6-v2"  # Must match indexer.py exactly


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

    def retrieve(self, query: str):
        """
        Search for the most relevant clinic documents for a given query.

        Returns a list of result dicts (text, score, source),
        or None if no document scores above the threshold.
        """
        # Embed the query — same model as indexing
        q_vector = self.model.encode([query], normalize_embeddings=True)
        q_vector = np.array(q_vector).astype("float32")

        # Search FAISS for top-K matches
        scores, indices = self.index.search(q_vector, TOP_K)

        # Safety check — if best score is too low, nothing is relevant
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
