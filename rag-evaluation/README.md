# RAG Evaluation Demo (Simple Python Example)

This project demonstrates the core idea behind evaluating a RAG system.

It explains the four key metrics:

- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

---

## Why This Matters

Without evaluation, a RAG system is guesswork.

Every change you make:
- chunk size
- embedding model
- retrieval logic
- prompt

can improve or break your system.

Evaluation tells you what changed.

---

## What This Demo Does

This script simulates:

- a user query
- retrieved documents
- a generated answer
- a ground truth answer

Then computes 4 evaluation metrics.

---

## Metrics Explained

### 1. Faithfulness
Is the answer supported by retrieved documents?

Low score → hallucination

---

### 2. Answer Relevance
Does the answer actually address the question?

Low score → wrong direction

---

### 3. Context Precision
How many retrieved documents are actually useful?

Low score → too much noise

---

### 4. Context Recall
Did we retrieve the necessary information?

Low score → missing key data

---

## How to Run

```bash
python rag_evaluation_demo.py
