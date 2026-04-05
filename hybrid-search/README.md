# Hybrid Search in Python: Vector Search + BM25 + RRF

This project demonstrates a simple end-to-end implementation of **Hybrid Search** in Python.

Hybrid search combines:

- **Vector search** for semantic similarity
- **Keyword search (BM25)** for exact term matching
- **Reciprocal Rank Fusion (RRF)** to merge both result lists into one final ranking

This is useful when user queries may contain both:
- meaning-based intent
- exact identifiers such as error codes, order numbers, version numbers, or product names

---

## Why Hybrid Search?

Vector search is very good at understanding meaning, but it may miss exact terms such as:

- `GlueException 0042`
- `ORD-99182`
- `v2.1.3`

Keyword search has the opposite strength:
- it is excellent at exact matches
- but it does not understand semantic similarity

Hybrid search combines both approaches so that the system can retrieve:
- semantically relevant documents
- exact-match documents

---

## How It Works

The pipeline has three parts:

1. **Vector Search**
   - Convert documents into embeddings using `sentence-transformers`
   - Store them in a FAISS index
   - Search by semantic similarity

2. **Keyword Search**
   - Tokenize the same documents
   - Score them with BM25 using `rank-bm25`

3. **Merge with RRF**
   - Ignore raw scores
   - Combine results based on rank positions only

---

## Project Structure

```text
.
├── requirements.txt
├── README.md
└── hybrid_search.py
