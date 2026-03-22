# RAG Hallucination Demo — Medical Clinic FAQ Bot

A focused demonstration of **how RAG eliminates LLM hallucination** in a
medical context. Each patient question is answered twice — once without RAG
(raw LLM, may hallucinate) and once with RAG (grounded from actual clinic
documents) — so the difference is visible side by side.

Built as part of the *"From Tokens to Agents"* AI learning series.
Series 2 — RAG & Semantic Search · Article 11: Why your RAG pipeline is
hallucinating — and how to fix it.

---

## Why this example exists

In the `rag-customer-support` example, we built a RAG pipeline and it worked.
But we never showed what happens **without** RAG.

This example answers that question directly:

> What does a hallucinated answer actually look like?
> How is a RAG-grounded answer different?
> Why does it matter in a medical context specifically?

The output of `app.py` shows both answers for every question so you can see
the difference with your own eyes.

---

## Why medical context?

Hallucination in a medical context is the clearest way to show why this problem
is dangerous — not just inconvenient.

A general-purpose LLM might answer "How much paracetamol can I take?" with a
plausible-sounding but wrong dose. In a retail context that is a bad experience.
In a medical context that is a patient safety risk.

RAG grounds the answer in the actual clinic's documented guidelines — the exact
dosage, the exact contraindications, the exact threshold for seeking emergency
care. There is no ambiguity and no guessing.

---

## What hallucination actually is

An LLM does not know the difference between what it knows confidently and what
it is making up. It generates the next most probable word — always. When it has
no relevant knowledge, it generates plausible-sounding text based on patterns
it has seen. That text is often wrong in ways that are hard to detect because
it sounds authoritative.

```
WITHOUT RAG
───────────────────────────────────────────────────────
Patient asks  →  LLM searches frozen training weights
                         ↓
              Finds general medical patterns
                         ↓
              Generates plausible-sounding answer
                         ↓
              May be wrong for your specific clinic's policy

WITH RAG
───────────────────────────────────────────────────────
Patient asks  →  Retrieve your actual clinic guidelines
                         ↓
              Inject guidelines into the prompt
                         ↓
              LLM reads YOUR documents and answers from them
                         ↓
              Answer is grounded in your real policies
```

---

## Project structure

```
rag-hallucination-demo/
├── indexer.py         # Phase A — embed clinic documents, save index to disk
├── retriever.py       # Semantic search over the saved index
├── app.py             # Phase B — runs each question WITH and WITHOUT RAG
├── requirements.txt   # Python dependencies
└── README.md
```

---

## File-by-file explanation

---

### indexer.py — Phase A: Build the index

Same pattern as every example in this repo. Runs once. Embeds documents and
saves two files to disk: `index.faiss` and `documents.json`.

#### The documents

```python
documents = [
    {
        "id": "doc_1",
        "text": "Paracetamol Dosage Policy: Adult patients may take 500mg to 1000mg..."
    },
    ...
]
```

Six clinic policy documents covering:
- Paracetamol dosage for adults
- Post-surgery wound care instructions
- Appointment cancellation policy and fees
- Antibiotic usage guidelines
- Fasting requirements before blood tests
- Children's fever management and dosing

These documents contain **specific** information — exact doses, exact fees,
exact thresholds — that a general LLM would have to guess at or approximate.
That is precisely why hallucination is dangerous here and why RAG is critical.

#### Embedding and saving

```python
model      = SentenceTransformer("all-MiniLM-L6-v2")
texts      = [doc["text"] for doc in documents]
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "index.faiss")
with open("documents.json", "w") as f:
    json.dump(documents, f, indent=2)
```

Each document is converted to a 384-dimensional vector. All vectors are stored
in a FAISS `IndexFlatIP` (brute-force cosine similarity search). Both the index
and original document text are saved to disk so the retriever can load them
without re-running the embedding step.

---

### retriever.py — Semantic Search

Identical structure to `rag-customer-support`. The constants at the top are the
key tuning parameters:

```python
SCORE_THRESHOLD = 0.40
TOP_K           = 2
MODEL_NAME      = "all-MiniLM-L6-v2"
```

- `SCORE_THRESHOLD = 0.40` — if the best match scores below this, we return
  `None`. In a medical context this is especially important. Returning a
  fallback ("please speak with a medical professional") is always safer than
  passing low-quality, irrelevant context to the LLM and getting a fabricated
  medical answer.

- `TOP_K = 2` — retrieve the top 2 most relevant documents. For medical
  questions this is usually enough since the clinic guidelines are focused.

- `MODEL_NAME` — must be identical to what `indexer.py` uses. This is the
  same-model rule: index and query must use the same embedding model or
  similarity scores are meaningless.

---

### app.py — The Hallucination Comparison

This is the key file in this example. It contains two separate functions and
runs both for every question.

#### Function 1 — answer_without_rag

```python
def answer_without_rag(question: str) -> str:
    prompt = f"""You are a medical clinic assistant.
Answer the patient's question as helpfully as you can.

Patient question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

This sends the question **directly to Claude with no retrieved context**.
The LLM has no knowledge of this clinic's specific policies. It answers from
its general training — producing answers that may be:
- Generically correct but not specific to this clinic
- Wrong about specific doses, fees, or thresholds
- Confidently stated regardless of accuracy

This is hallucination in a real-world setting.

#### Function 2 — answer_with_rag

```python
def answer_with_rag(question: str) -> str:
    results = retriever.retrieve(question)

    if results is None:
        return "I don't have specific clinic guidelines for that..."

    context = "\n\n".join([
        f"[Clinic Document {i+1} | Relevance: {r['score']}]\n{r['text']}"
        for i, r in enumerate(results)
    ])

    prompt = f"""You are a medical clinic assistant.
Answer the patient's question using ONLY the clinic guidelines provided below.
Be clear and precise — this is a medical context where accuracy is critical.
Never guess or estimate medical information.

=== CLINIC GUIDELINES ===
{context}

=== PATIENT QUESTION ===
{question}

=== YOUR ANSWER ==="""
```

Two critical differences from `answer_without_rag`:

1. **Retrieval first** — we fetch the actual clinic guidelines before calling Claude
2. **"ONLY the clinic guidelines"** — we explicitly instruct Claude not to use
   its own training knowledge. It must answer from our documents.

The phrase "Never guess or estimate medical information" further reinforces that
in this domain, saying "I don't know" is always better than hallucinating.

#### The run_comparison function

```python
def run_comparison(question: str):
    print(f"PATIENT QUESTION: {question}")

    print("[ WITHOUT RAG — raw LLM, no clinic context ]")
    print(answer_without_rag(question))

    print("[ WITH RAG — grounded from clinic documents ]")
    print(answer_with_rag(question))
```

Runs both functions and prints the results side by side. This is the core of
the demonstration — seeing the two answers next to each other makes the
difference impossible to miss.

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here   # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-your-key-here      # Windows
```

---

## Running the project

### Step 1 — Build the index (run once)

```bash
python indexer.py
```

Expected output:
```
Loading embedding model...
Embedding 6 clinic documents...
Building FAISS index...
Saved → index.faiss
Saved → documents.json

Indexing complete. 6 documents indexed and ready.
```

### Step 2 — Run the hallucination comparison

```bash
python app.py
```

Expected output structure:
```
RAG HALLUCINATION DEMO — Medical Clinic FAQ Bot
Each question is answered twice:
  1. WITHOUT RAG — raw LLM (may hallucinate specific details)
  2. WITH RAG    — grounded from actual clinic documents

=================================================================
PATIENT QUESTION: How much paracetamol can I take in a day?
=================================================================

[ WITHOUT RAG — raw LLM, no clinic context ]
-----------------------------------------------------------------
The generally recommended maximum dose of paracetamol for adults
is 4 grams (4000mg) per day, taken as 1-2 tablets (500-1000mg)
every 4-6 hours...
(answer may be generic, may miss the liver disease caveat,
 may not match this clinic's specific guidelines)

[ WITH RAG — grounded from clinic documents ]
-----------------------------------------------------------------
According to our clinic guidelines, adults may take 500mg to
1000mg of paracetamol every 4 to 6 hours, with a maximum of
4000mg (4g) in 24 hours. Important: if you have liver disease,
your limit is 2000mg per day. Do not combine with other
paracetamol-containing medications as overdose can cause
severe liver damage.
(answer is precise, cites contraindications, matches the policy exactly)
```

---

## Key concepts this example teaches

| Concept | Where to look |
|---|---|
| What hallucination looks like | `answer_without_rag()` output in `app.py` |
| How RAG eliminates it | `answer_with_rag()` output vs without |
| Why "ONLY from context" matters | The instruction in the augmented prompt |
| Safe fallback pattern | `if results is None` returning a safe message |
| Why score threshold matters in medical context | `SCORE_THRESHOLD` comment in `retriever.py` |
| Side-by-side comparison pattern | `run_comparison()` in `app.py` |

---

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `No such file: index.faiss` | `indexer.py` not run yet | Run `python indexer.py` first |
| `TypeError: module object is not callable` | Import naming conflict | Use `from retriever import Retriever as RetrieverClass` |
| `credit balance is too low` | No Anthropic credits | Add credits at console.anthropic.com/settings/billing |
| `Could not resolve authentication` | API key not set | `export ANTHROPIC_API_KEY=your-key` |

---

## Extending this example

### Add your own medical documents
Edit the `documents` list in `indexer.py` and re-run `python indexer.py`.
No other files need to change.

### Test with questions outside the knowledge base
Try asking something the clinic documents don't cover — for example:
`"What is the treatment for appendicitis?"`. With RAG, the score threshold
will kick in and return the safe fallback message. Without RAG, the LLM
will answer from general medical knowledge — demonstrating hallucination risk
for out-of-scope questions.

### Raise the score threshold for stricter grounding
In `retriever.py`, increase `SCORE_THRESHOLD` from 0.40 to 0.55. The RAG
bot will answer fewer questions but everything it answers will be more
precisely grounded in the actual documents.

---

## Dependencies

```
anthropic>=0.40.0             # Anthropic Python SDK — calls the Claude API
sentence-transformers>=3.0.0  # Embedding model runs locally
faiss-cpu>=1.8.0              # Vector similarity search
numpy>=1.26.0                 # Array operations for FAISS
```
