from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# -----------------------------
# Sample documents
# -----------------------------
documents = [
    "Glue job failed with OOM error",
    "How to fix GlueException 0042",
    "Retry strategy for AWS Glue jobs",
    "Understanding memory issues in Spark",
    "Error code 0042 occurs due to timeout"
]

# -----------------------------
# 1. Embeddings (Vector Search)
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# -----------------------------
# 2. Keyword Search (BM25)
# -----------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# -----------------------------
# 3. Query
# -----------------------------
query = "GlueException 0042"
query_embedding = model.encode([query])

# -----------------------------
# 4. Vector Search
# -----------------------------
k = 5
D, I = index.search(np.array(query_embedding), k)

vector_results = I[0]  # indices of top docs

# -----------------------------
# 5. Keyword Search
# -----------------------------
tokenized_query = query.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)

keyword_results = np.argsort(bm25_scores)[::-1][:k]

# -----------------------------
# 6. RRF (Reciprocal Rank Fusion)
# -----------------------------
def rrf_score(rank, k=60):
    return 1 / (k + rank)

rrf_scores = {}

# Add vector ranks
for rank, doc_id in enumerate(vector_results):
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score(rank)

# Add keyword ranks
for rank, doc_id in enumerate(keyword_results):
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score(rank)

# -----------------------------
# 7. Final Ranking
# -----------------------------
final_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# -----------------------------
# 8. Print Results
# -----------------------------
print("\n🔍 Hybrid Search Results:\n")
for doc_id, score in final_results:
    print(f"Score: {score:.4f} | Document: {documents[doc_id]}")
