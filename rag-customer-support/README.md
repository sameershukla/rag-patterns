# RAG Customer Support Bot

A minimal, framework-free implementation of **Retrieval Augmented Generation (RAG)**
built from scratch in pure Python. No LangChain. No abstractions. Every line is intentional
so you understand exactly what is happening at each step.

Built as part of the *"From Tokens to Agents"* AI learning series.
Series 2 — RAG & Semantic Search · Article 10: Building RAG from scratch in pure Python.

---

## What problem does this solve?

A standard LLM like Claude has no knowledge of your company's internal policies.
If a customer asks "Can I return a broken item?", Claude will answer from its general
training — which may be vague, wrong, or hallucinated.

This project solves that by implementing RAG:
1. We store our actual policy documents in a searchable vector index
2. When a customer asks a question, we find the most relevant policy
3. We inject that policy into the prompt before Claude answers
4. Claude reads our real policy and gives a grounded, accurate answer

---

## Project structure

```
rag-customer-support/
├── indexer.py         # Phase A — run once to build and save the FAISS index
├── retriever.py       # Loads the saved index, handles semantic search
├── app.py             # Phase B — query pipeline, answers customer questions
├── requirements.txt   # Python dependencies
└── README.md
```

### Why three files instead of one?

| File | When it runs | How often |
|---|---|---|
| `indexer.py` | When documents are added or updated | Once, or on a schedule |
| `retriever.py` | Loaded once at app startup | Stays in memory |
| `app.py` | On every customer question | Every single request |

If indexing and querying were in one file, you would rebuild the entire vector
index on every customer question. That would take seconds per request and defeats
the entire purpose of pre-building an index.

---

## The two phases of RAG

RAG is not one pipeline — it is two pipelines running at completely different times.

```
PHASE A — INDEXING (offline, run once)
────────────────────────────────────────────────────────────
Documents → Embed → Store in FAISS → Save to disk
                                           ↓
                                     index.faiss
                                   documents.json

PHASE B — QUERYING (online, runs per request)
────────────────────────────────────────────────────────────
Customer question → Embed → Search FAISS → Top-K chunks
                                                 ↓
                             Inject into prompt + Send to Claude
                                                 ↓
                                         Grounded answer
```

---

## File-by-file explanation

---

### indexer.py — Phase A: Build the index

This file runs once. It reads your documents, converts them to vectors,
and saves everything to disk so the query pipeline can load it instantly.

#### The documents

```python
documents = [
    {
        "id": "doc_1",
        "text": "Return Policy: Customers can return any item within 30 days...",
    },
    ...
]
```

Each document is a plain Python dict with two fields:
- `id` — a unique identifier for the document
- `text` — the raw policy text that will be embedded and searched

In a real system, these would be loaded from Confluence, a database, or S3.
For learning, we hardcode them directly.

#### Loading the embedding model

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

This loads a pre-trained embedding model from HuggingFace.
`all-MiniLM-L6-v2` is a lightweight model that converts text into
384-dimensional vectors. It is fast, accurate, and free to use locally.

The first time you run this, it downloads the model (~90MB).
After that it is cached and loads instantly.

#### Embedding the documents

```python
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")
```

Line by line:
- `texts` — extracts just the text string from each document dict
- `model.encode()` — converts each text string into a vector of 384 numbers
- `normalize_embeddings=True` — scales every vector to length 1, which is
  required for cosine similarity to work correctly with FAISS IndexFlatIP
- `.astype("float32")` — FAISS requires 32-bit floats, not 64-bit

After this step, `embeddings` is a 2D array of shape (5, 384) —
5 documents, each represented as 384 numbers.

#### Building the FAISS index

```python
dim = embeddings.shape[1]       # 384
index = faiss.IndexFlatIP(dim)  # Inner Product index
index.add(embeddings)
```

- `dim` — the number of dimensions in each vector (384 for this model)
- `IndexFlatIP` — "Flat" means it compares every query vector against every
  stored vector (brute force). "IP" means Inner Product, which equals cosine
  similarity when vectors are normalised. For 5 documents this is instant.
  For millions of documents you would switch to `IndexHNSWFlat` for speed.
- `index.add(embeddings)` — loads all 5 document vectors into the index

#### Saving to disk

```python
faiss.write_index(index, "index.faiss")

with open("documents.json", "w") as f:
    json.dump(documents, f, indent=2)
```

Two files are saved:
- `index.faiss` — the FAISS vector index (the numbers and search structure)
- `documents.json` — the original document text and metadata

FAISS only stores vectors — it has no knowledge of what text those vectors
came from. We save `documents.json` separately so the retriever can map
a FAISS result index (e.g. index 2) back to the actual document text.

---

### retriever.py — Semantic Search

This file is a reusable class that loads the pre-built index from disk
and handles all search logic. It is imported and used by `app.py`.

#### Constants at the top

```python
SCORE_THRESHOLD = 0.4
TOP_K           = 2
MODEL_NAME      = "all-MiniLM-L6-v2"
```

These are defined as constants (uppercase) rather than magic numbers buried
in the code. This makes them easy to find and tune:
- `SCORE_THRESHOLD` — minimum similarity score to consider a document relevant.
  Below this, we return None instead of retrieving low-quality chunks.
- `TOP_K` — how many chunks to retrieve per query. 2 is enough for short
  policy docs. For longer documents you might use 3–5.
- `MODEL_NAME` — must be identical to what `indexer.py` used. This is the
  most critical rule in all of RAG — same model for indexing and querying.

#### The Retriever class — __init__

```python
def __init__(self):
    self.model = SentenceTransformer(MODEL_NAME)
    self.index = faiss.read_index("index.faiss")
    with open("documents.json", "r") as f:
        self.documents = json.load(f)
```

When `Retriever()` is called in `app.py`, this runs once at startup:
- Loads the embedding model into memory
- Loads the pre-built FAISS index from `index.faiss`
- Loads the document metadata from `documents.json`

All three are stored as `self.` attributes so they stay in memory and
are reused across every call to `retrieve()` — no reloading per query.

#### The retrieve method

```python
def retrieve(self, query: str) -> list[dict] | None:
```

Takes a plain text question. Returns either:
- A list of dicts (each with `text`, `score`, `source`) if relevant docs are found
- `None` if no document exceeds the score threshold

**Step 1 — Embed the query**
```python
q_vector = self.model.encode([query], normalize_embeddings=True)
q_vector = np.array(q_vector).astype("float32")
```

The query goes through the exact same embedding model as the documents.
This is what makes similarity comparison valid — both the query vector and
document vectors live in the same 384-dimensional space.

**Step 2 — Search FAISS**
```python
scores, indices = self.index.search(q_vector, TOP_K)
```

FAISS compares the query vector against all stored document vectors and
returns the TOP_K closest matches. `scores` contains the similarity values
(0 to 1, higher is more similar). `indices` contains the position of each
matching document in the original embeddings array.

**Step 3 — Score threshold check**
```python
best_score = float(scores[0][0])
if best_score < SCORE_THRESHOLD:
    return None
```

This is a critical production safety check. FAISS always returns TOP_K
results — even if none of them are actually relevant to the query. Without
this check, a completely unrelated question like "What is the weather today?"
would still retrieve your return policy with a low score, and Claude would
try to answer using that irrelevant policy text. Returning None tells `app.py`
to respond with a graceful fallback message instead.

**Step 4 — Build and return results**
```python
results = []
for i, idx in enumerate(indices[0]):
    results.append({
        "text":   self.documents[idx]["text"],
        "score":  round(float(scores[0][i]), 2),
        "source": self.documents[idx]["id"],
    })
return results
```

`indices[0]` contains the position numbers of the matched documents.
We use those positions to look up the actual text and metadata from
`self.documents`. The score is rounded to 2 decimal places for clean
display in the prompt.

---

### app.py — Phase B: The Query Pipeline

This is the live, user-facing side of the system. It runs on every
customer question. It ties together the retriever and Claude.

#### Setup — runs once at startup

```python
retriever = Retriever()
client    = anthropic.Anthropic()
```

Both objects are created once when `app.py` starts, not on every question.
Creating them on every question would reload the model and index from disk
on every request — very slow.

#### The answer_question function

**Step 1 — Retrieve**
```python
results = retriever.retrieve(question)

if results is None:
    return "I don't have information about that..."
```

Ask the retriever to find relevant documents. If it returns None (no good match
found), we return a fallback message immediately — no Claude call needed.
This saves API cost and prevents hallucination.

**Step 2 — Build context**
```python
context = "\n\n".join([
    f"[Policy {i+1} | Relevance: {r['score']}]\n{r['text']}"
    for i, r in enumerate(results)
])
```

The retrieved documents are formatted into a readable context block. Including
the relevance score helps Claude (and you when debugging) understand how
confident the retrieval was. Including the policy number makes it easy to
trace which document each part of the answer came from.

Example of what `context` looks like:
```
[Policy 1 | Relevance: 0.72]
Damaged Items: If you receive a damaged or defective item, contact us within
48 hours of delivery...

[Policy 2 | Relevance: 0.51]
Return Policy: Customers can return any item within 30 days of purchase...
```

**Step 3 — Build the augmented prompt**
```python
prompt = f"""You are a helpful customer support agent for an e-commerce store.
Answer the customer's question using ONLY the policy information provided below.
...
=== STORE POLICIES ===
{context}

=== CUSTOMER QUESTION ===
{question}

=== YOUR ANSWER ==="""
```

This is the augmentation step — the core of RAG. The prompt has three sections:
- System instruction — tells Claude its role and constraints
- Retrieved context — the actual policy documents injected at runtime
- The customer's question

The phrase "using ONLY the policy information provided" is critical.
It prevents Claude from mixing in its own general knowledge, which could
contradict your specific policies.

**Step 4 — Generate with Claude**
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}]
)
return response.content[0].text
```

The augmented prompt is sent to Claude. Claude reads the policy context and
generates a grounded answer from it. `max_tokens=300` limits the response
length — enough for a clear support answer without unnecessary padding.
`response.content[0].text` extracts just the text from the API response object.

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

Get your API key at: https://console.anthropic.com → Settings → API Keys

---

## Running the project

### Step 1 — Build the index (run once)

```bash
python indexer.py
```

Expected output:
```
Loading embedding model...
Embedding 5 documents...
Building FAISS index...
Saved index.faiss
Saved documents.json

Indexing complete. 5 documents indexed.
```

This creates two files in your project folder:
- `index.faiss` — the vector index
- `documents.json` — the document text and metadata

Only re-run `indexer.py` when your documents change.

### Step 2 — Answer customer questions

```bash
python app.py
```

Expected output:
```
Customer: My package arrived broken. What do I do?
Bot:      I'm sorry to hear your package arrived damaged! Please contact
          us at support@store.com within 48 hours of delivery with a photo
          of the damage. We will ship a replacement at no charge within
          3 business days.
------------------------------------------------------------
Customer: How long does free shipping take?
Bot:      Free shipping is available on all orders over $50 and takes
          5-7 business days to arrive.
------------------------------------------------------------
Customer: Can I cancel my order from 3 hours ago?
Bot:      Unfortunately, orders can only be cancelled within 1 hour of
          placement. Since your order is now in processing, it can no longer
          be cancelled. Once you receive it, you can return it within 30 days
          under our return policy.
------------------------------------------------------------
```

---

## Key concepts this code teaches

| Concept | Where to look |
|---|---|
| Why RAG exists | The `documents` list in `indexer.py` — LLM has no knowledge of these |
| Phase A vs Phase B split | `indexer.py` (offline) vs `app.py` (online) |
| Embedding text to vectors | `model.encode()` in `indexer.py` |
| Normalisation for cosine similarity | `normalize_embeddings=True` in both files |
| Saving and loading a FAISS index | `faiss.write_index` / `faiss.read_index` |
| The same model rule | `MODEL_NAME` constant shared between both files |
| Score threshold safety guard | `if best_score < SCORE_THRESHOLD` in `retriever.py` |
| Prompt augmentation | `=== STORE POLICIES ===` block injected in `app.py` |
| Grounding the LLM | "Answer using ONLY the policy information" instruction |

---

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `No such file or directory: index.faiss` | `indexer.py` was never run | Run `python indexer.py` first |
| `TypeError: module object is not callable` | Import naming conflict | Use `from retriever import Retriever as RetrieverClass` |
| `credit balance is too low` | No Anthropic credits | Add credits at console.anthropic.com/settings/billing |
| `Could not resolve authentication method` | API key not set | `export ANTHROPIC_API_KEY=your-key` |
| Poor answer quality | Wrong chunks being retrieved | Tune `SCORE_THRESHOLD` and `TOP_K` in `retriever.py` |

---

## Extending this project

### Add your own documents
Edit the `documents` list in `indexer.py`, re-run `python indexer.py`.
No changes needed anywhere else.

### Tune retrieval quality
In `retriever.py`:
- Raise `SCORE_THRESHOLD` (e.g. 0.55) — stricter, fewer answers, less hallucination risk
- Lower `SCORE_THRESHOLD` (e.g. 0.30) — more permissive, answers more questions
- Increase `TOP_K` (e.g. 4) — more context per answer, useful for complex questions

### Swap FAISS for a production vector database
Only `retriever.py` changes. `indexer.py` and `app.py` are completely unaffected.
Options: Pinecone, Chroma, Weaviate, Snowflake Cortex Search.

---

## Dependencies explained

```
anthropic>=0.40.0             # Anthropic Python SDK — calls the Claude API
sentence-transformers>=3.0.0  # Loads and runs the embedding model locally
faiss-cpu>=1.8.0              # Vector index — fast similarity search on CPU
numpy>=1.26.0                 # Array operations required by FAISS
```
