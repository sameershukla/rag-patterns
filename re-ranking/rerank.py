from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch


# -----------------------------------
# Sample documents
# -----------------------------------
documents = [
    "AWS Glue jobs can fail with out of memory errors when partitions are too large.",
    "GlueException 0042 happens when the job configuration is invalid.",
    "Retry strategies help improve reliability in distributed systems.",
    "To fix GlueException 0042, check IAM permissions and job parameters.",
    "Spark memory tuning can reduce executor failures."
]

query = "How to fix GlueException 0042"


# -----------------------------------
# Step 1: Fast retrieval using bi-encoder
# -----------------------------------
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = bi_encoder.encode(documents, convert_to_tensor=True)
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)

# cosine similarity between query and all documents
scores = util.cos_sim(query_embedding, doc_embeddings)[0]

top_k = 5
top_results = torch.topk(scores, k=top_k)

retrieved_docs = []
print("\nTop documents from bi-encoder retrieval:\n")
for rank, (score, doc_idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
    doc = documents[doc_idx]
    retrieved_docs.append((doc_idx.item(), doc, score.item()))
    print(f"{rank}. Score: {score:.4f} | {doc}")


# -----------------------------------
# Step 2: Re-ranking using cross-encoder
# -----------------------------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# create query-document pairs
pairs = [[query, doc] for _, doc, _ in retrieved_docs]

# cross-encoder gives a refined relevance score
rerank_scores = cross_encoder.predict(pairs)

reranked_results = []
for (doc_idx, doc, retrieval_score), rerank_score in zip(retrieved_docs, rerank_scores):
    reranked_results.append({
        "doc_id": doc_idx,
        "document": doc,
        "retrieval_score": retrieval_score,
        "rerank_score": float(rerank_score)
    })

# sort by re-ranker score descending
reranked_results = sorted(reranked_results, key=lambda x: x["rerank_score"], reverse=True)


# -----------------------------------
# Step 3: Final top documents for LLM
# -----------------------------------
final_top_n = 3

print("\nRe-ranked documents:\n")
for rank, item in enumerate(reranked_results, start=1):
    print(
        f"{rank}. Rerank Score: {item['rerank_score']:.4f} "
        f"| Retrieval Score: {item['retrieval_score']:.4f} "
        f"| {item['document']}"
    )

print(f"\nTop {final_top_n} documents to send to the LLM:\n")
for rank, item in enumerate(reranked_results[:final_top_n], start=1):
    print(f"{rank}. {item['document']}")
