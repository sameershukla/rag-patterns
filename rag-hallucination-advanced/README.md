# RAG Hallucination Advanced — Phase 2 Grounding Techniques

An upgraded version of the Medical Clinic FAQ Bot that applies all four
Phase 2 hallucination control techniques in working code. The basic version
showed what hallucination looks like. This version shows how to systematically
eliminate it.

Built as part of the *"From Tokens to Agents"* AI learning series.
Series 2 — RAG & Semantic Search · Article 11: Why your RAG pipeline is
hallucinating — and how to fix it.

---

## What is new compared to the basic hallucination demo

| Feature | Basic demo | This demo |
|---|---|---|
| Score threshold guard | Yes | Yes |
| Citation requirement | No | Yes — LLM must cite [Doc 1], [Doc 2] |
| Confidence scoring | No | Yes — LLM rates answer 1-5 |
| Confidence gate | No | Yes — fallback if confidence < 3 |
| Score logging | No | Yes — every query logged to rag_logs.jsonl |
| Golden eval set | No | Yes — 10 questions, measurable accuracy |

---

## Project structure

```
rag-hallucination-advanced/
├── indexer.py         # Phase A — embed clinic docs, save index to disk
├── retriever.py       # Semantic search + score logging to rag_logs.jsonl
├── app.py             # Phase B — query pipeline with all 4 grounding techniques
├── eval.py            # Golden evaluation set — measures accuracy rate
├── requirements.txt
└── README.md
```

---

## The four Phase 2 grounding techniques — in code

---

### Technique 1 — Explicit "I don't know" instruction

The basic prompt says "answer only from context." The advanced prompt tells
the LLM exactly what to do when the answer is not in the context:

```python
"If the answer is NOT explicitly in the guidelines, do not guess.
 Instead write: 'This is not covered in our current guidelines.
 Please speak with a medical professional.'"
```

Without this, the LLM fills the gap. With it, it returns a safe fallback.

---

### Technique 2 — Citation requirement

Every factual claim must be cited with the document it came from:

```python
"After every factual claim, cite the source using [Doc 1] or [Doc 2]."
```

Example answer the LLM produces:
```
The maximum daily dose is 4000mg [Doc 1]. Patients with liver disease
must not exceed 2000mg per day [Doc 1].
```

This technique eliminates extrinsic hallucination — the LLM cannot add
information it cannot attribute to a retrieved document.

---

### Technique 3 — Confidence scoring

The LLM rates its own answer before giving it:

```python
"Respond in this exact format:
CONFIDENCE: [1-5 where 5=fully supported by the docs, 1=not in the docs]
ANSWER: [your cited answer]"
```

The `parse_response()` function in `app.py` extracts the confidence score.
If confidence is below `MIN_CONFIDENCE = 3`, a safe fallback is returned
instead of the answer — regardless of what the answer says.

---

### Technique 4 — Context-first prompt structure

Context goes before the question. This counteracts the "lost in the middle"
problem where LLMs pay less attention to content in the middle of a long prompt:

```python
prompt = f"""You are a medical clinic assistant.

=== CLINIC GUIDELINES ===      ← context FIRST
{context}

=== PATIENT QUESTION ===       ← question LAST
{question}

=== INSTRUCTIONS ===
..."""
```

---

## The golden evaluation set (eval.py)

10 questions with known correct answers. Each question has a `must_contain`
list of terms that must appear in the answer for it to pass. One question
(`q10`) has no answer in the docs — the system must return a fallback.

```
q01 — max daily paracetamol dose        → must contain "4000" and "4g"
q02 — keep wound dry for how long       → must contain "48"
q03 — cancellation fee under 24 hours   → must contain "50"
q04 — fast before blood test            → must contain "8" and "12"
q05 — can child take aspirin            → must contain "16"
q06 — amoxicillin frequency             → must contain "three" or "3"
q07 — fever threshold after surgery     → must contain "38.5"
q08 — blood test days                   → must contain "monday" and "friday"
q09 — paracetamol limit for liver dis.  → must contain "2000"
q10 — broken leg treatment (off-topic)  → must return fallback
```

Run the eval after every change. Track the score over time. This is the
only way to know if a change actually improved your system.

---

## Reading rag_logs.jsonl

Every query is logged automatically by `retriever.py`:

```json
{"timestamp": "2025-11-14T09:32:11", "query": "How much paracetamol...",
 "best_score": 0.812, "answered": true, "hallucination_risk": false}
{"timestamp": "2025-11-14T09:32:18", "query": "What is the weather...",
 "best_score": 0.198, "answered": false, "hallucination_risk": true}
```

Fields:
- `best_score` — highest similarity score from retrieval
- `answered` — true if score was above SCORE_THRESHOLD (0.40)
- `hallucination_risk` — true if score was below RISK_THRESHOLD (0.60)

Use this log to spot patterns:
- Many `hallucination_risk: true` entries → knowledge base has gaps
- Average scores dropping over time → knowledge base is becoming stale
- Specific queries consistently scoring low → add better documents

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## Running the project

### Step 1 — Build the index (run once)

```bash
python indexer.py
```

### Step 2 — Run the demo

```bash
python app.py
```

Expected output structure:
```
─────────────────────────────────────────────────────────────────
Q: How much paracetamol can I take in a day?
Confidence: [████░] 4/5
Sources: doc_1

A: Adults may take 500mg to 1000mg every 4 to 6 hours [Doc 1].
   The maximum daily dose must not exceed 4000mg (4g) in 24 hours [Doc 1].
   If you have liver disease, your limit is 2000mg per day [Doc 1].

─────────────────────────────────────────────────────────────────
Q: What is the weather forecast for tomorrow?
Confidence: [░░░░░] 0/5  [FALLBACK]
Sources: none

A: I don't have information about that in our clinic guidelines.
   Please speak directly with one of our medical staff.
```

### Step 3 — Measure your accuracy rate

```bash
python eval.py
```

Expected output:
```
=================================================================
RUNNING GOLDEN EVALUATION SET
Total questions: 10
=================================================================
✓ [q01] conf=5/5 | all required terms found
✓ [q02] conf=5/5 | all required terms found
...
=================================================================
EVAL COMPLETE
Score:  90%  (9 passed / 1 failed out of 10)
Production ready — score above 90%
=================================================================
```

---

## What to do when eval score is low

| Score | Diagnosis | Where to look |
|---|---|---|
| q01-q09 failing | Wrong chunks retrieved | Check SCORE_THRESHOLD, check document text quality |
| q10 failing | Fallback not triggering | SCORE_THRESHOLD too low, or model ignoring instructions |
| Confidence always low | Prompt structure issue | Adjust context-first order, sharpen instructions |
| Scores dropping over time | Stale knowledge base | Re-run indexer.py with updated documents |

---

## Key concepts this code teaches

| Concept | Where to look |
|---|---|
| Citation requirement | `prompt` in `app.py` — "cite the source using [Doc 1]" |
| Confidence scoring | `prompt` format + `parse_response()` in `app.py` |
| Confidence gate | `if confidence < MIN_CONFIDENCE` in `app.py` |
| Context-first structure | Prompt order — context before question in `app.py` |
| Score threshold logging | `log_entry` in `retriever.py` |
| Hallucination risk flag | `RISK_THRESHOLD` in `retriever.py` |
| Golden eval set | `eval_set` list in `eval.py` |
| must_contain checking | `run_eval()` loop in `eval.py` |

---

## Dependencies

```
anthropic>=0.40.0             # Anthropic Python SDK
sentence-transformers>=3.0.0  # Local embedding model
faiss-cpu>=1.8.0              # Vector similarity search
numpy>=1.26.0                 # Array operations
```
