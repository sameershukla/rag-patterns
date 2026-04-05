# Query Transformation in Python

This project demonstrates a simple implementation of **query transformation**.

Query transformation means improving a user's query before retrieval so that search works better.

Instead of sending the raw query directly to a search system, we first:
- rewrite it
- expand it
- generate alternate versions

This helps retrieval systems find more relevant documents.

---

## Why Query Transformation Is Needed

Users often type:
- short queries
- vague queries
- incomplete phrases
- inconsistent terminology

Examples:

- `why glue fails`
- `glue oom`
- `retry issue`

These are understandable to humans, but not always ideal for retrieval systems.

Query transformation improves them before search.

---

## What This Example Covers

This demo includes three simple techniques:

### 1. Query Rewriting
Turn a vague or incomplete query into a clearer search query.

Example:

- Input: `glue oom`
- Rewritten: `AWS Glue out of memory error causes, memory tuning, partition sizing, Spark executor memory issues`

### 2. Query Expansion
Add related terms or synonyms to improve recall.

Example:

- `oom` → `out of memory`, `memory error`
- `glue` → `aws glue`, `glue job`

### 3. Multi-Query Generation
Generate multiple versions of the same query.

Example:

- `How do I troubleshoot AWS Glue out of memory errors?`
- `Best way to fix AWS Glue out of memory errors`

---

## Project Files

```text
.
├── query_transformation.py
├── requirements.txt
└── README.md
