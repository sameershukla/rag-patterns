"""
indexer.py — Phase A: Indexing
Run this once to embed your documents and save the FAISS index to disk.
Re-run only when documents change.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Documents (your knowledge base) ──────────────────────────────────────────

documents = [
    {
        "id": "doc_1",
        "text": "Return Policy: Customers can return any item within 30 days of purchase. "
                "Items must be unused and in original packaging. Refunds are processed "
                "within 5-7 business days back to the original payment method.",
    },
    {
        "id": "doc_2",
        "text": "Shipping Policy: Standard shipping takes 5-7 business days and costs $4.99. "
                "Express shipping takes 2-3 business days and costs $12.99. "
                "Free shipping is available on all orders over $50.",
    },
    {
        "id": "doc_3",
        "text": "Damaged Items: If you receive a damaged or defective item, contact us within "
                "48 hours of delivery. Send a photo of the damage to support@store.com. "
                "We will ship a replacement at no charge within 3 business days.",
    },
    {
        "id": "doc_4",
        "text": "Order Cancellation: Orders can be cancelled within 1 hour of placement. "
                "After 1 hour, the order enters processing and cannot be cancelled. "
                "You may still return the item once received using our return policy.",
    },
    {
        "id": "doc_5",
        "text": "Gift Cards: Gift cards never expire and can be used on any purchase. "
                "Gift cards cannot be exchanged for cash. Lost or stolen gift cards "
                "cannot be replaced unless the original purchase receipt is provided.",
    },
]

# ── Indexing Pipeline ─────────────────────────────────────────────────────────

def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {len(documents)} documents...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, "index.faiss")
    print("Saved index.faiss")

    # Save documents metadata to disk (so retriever can map index → text)
    with open("documents.json", "w") as f:
        json.dump(documents, f, indent=2)
    print("Saved documents.json")

    print(f"\nIndexing complete. {index.ntotal} documents indexed.")


if __name__ == "__main__":
    build_index()
