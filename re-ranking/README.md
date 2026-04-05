# Re-ranking in RAG: Bi-encoder + Cross-encoder

This project demonstrates a simple end-to-end example of **re-ranking** in a Retrieval-Augmented Generation (RAG) pipeline.

It uses two stages:

1. **Bi-encoder retrieval**  
   Fast retrieval using embeddings to find the top candidate documents

2. **Cross-encoder re-ranking**  
   A second, more accurate pass that re-orders the retrieved documents based on how well each document answers the query

---

## Why Re-ranking is Needed

Vector search is fast and good at finding similar documents, but it does not always place the best result at the top.

For example, the correct answer may be present in the top retrieved documents, but ranked lower than more general or loosely related results.

Re-ranking solves this problem by taking the top retrieved documents and evaluating them again more carefully.

---

## How It Works

### Step 1: Retrieve candidate documents
A **bi-encoder** converts the query and documents into embeddings separately.

This is fast and scalable because:
- document embeddings can be precomputed
- similarity search is efficient

### Step 2: Re-rank retrieved documents
A **cross-encoder** reads the query and each candidate document together.

This allows it to understand:
- exact relationship
- context
- whether the document directly answers the question

The cross-encoder then produces a new relevance score for each candidate.

### Step 3: Send best documents to the LLM
Take the **top 3 to 5 re-ranked documents** and pass only those to the LLM as final context.

---

## Bi-encoder vs Cross-encoder

| Aspect        | Bi-encoder (Retrieval)              | Cross-encoder (Re-ranking)           |
|--------------|-------------------------------------|--------------------------------------|
| How it works | Encodes query and doc separately     | Reads query and doc together         |
| Speed        | Very fast                            | Slower                               |
| Accuracy     | Good                                 | Better                               |
| Scale        | Millions of documents                | Small candidate sets only            |
| Use it for   | Broad retrieval                      | Final precision ranking              |

---

## Project Files

```text
.
├── rerank.py
├── requirements.txt
└── README.md
