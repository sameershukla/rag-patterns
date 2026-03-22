# LangChain RAG Demo — E-commerce Customer Support Bot

A complete, production-ready RAG pipeline built with LangChain (v0.3+).
The same e-commerce customer support bot from the scratch implementation
(Chapter 13), rebuilt using LangChain's LCEL (LangChain Expression Language).

Built as part of the *"From Tokens to Agents"* AI learning series.
Part 3 — Building LLM Applications · Chapter 26: Building a RAG pipeline
in LangChain.

---

## What this demonstrates

- **Phase A — Indexing** (`indexer.py`): Load documents → chunk with
  `RecursiveCharacterTextSplitter` → embed → save FAISS index to disk
- **Phase B — Querying** (`app.py`): Load index → LCEL chain →
  retriever + prompt + LLM + parser → streaming answer with memory
- **Conversational memory** using `RunnableWithMessageHistory` —
  the modern LangChain v0.3 memory pattern
- **Streaming** — one method change from `.invoke()` to `.stream()`
- **Session-based memory** — separate conversation history per user

---

## How this compares to building from scratch

| Component | From scratch (Ch 13) | LangChain (Ch 26) |
|---|---|---|
| Embedding | `SentenceTransformer.encode()` | `HuggingFaceEmbeddings` |
| Vector index | `faiss.IndexFlatIP` + manual `add()` | `FAISS.from_documents()` |
| Chunking | Manual `split()` | `RecursiveCharacterTextSplitter` |
| Retriever | Custom `Retriever` class | `vectorstore.as_retriever()` |
| Prompt | f-string | `ChatPromptTemplate` |
| LLM call | `anthropic.Anthropic()` | `ChatAnthropic` |
| Output parsing | `response.content[0].text` | `StrOutputParser()` |
| Memory | Manual messages list | `RunnableWithMessageHistory` |
| Streaming | `client.messages.stream()` | `.stream()` on any chain |

Same concepts. Less boilerplate.

---

## Project structure

```
rag-langchain/
├── indexer.py         # Phase A — run once to build and save the FAISS index
├── app.py             # Phase B — RAG chain with memory and streaming
├── requirements.txt   # Python dependencies
└── README.md
```

---

## File-by-file explanation

---

### indexer.py — Phase A: Build the index

**Run once.** Loads store policy documents, chunks them, embeds them using
`all-MiniLM-L6-v2`, and saves the FAISS index to a local `faiss_index/` folder.

#### Key components

**`RecursiveCharacterTextSplitter`**
LangChain's production-grade chunker. Unlike a manual `text.split()`, it tries
to split on paragraph boundaries first, then sentences, then words — preserving
semantic meaning at chunk boundaries.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # max characters per chunk
    chunk_overlap=30,   # overlap between consecutive chunks
)
```

`chunk_overlap` ensures no context is lost at chunk boundaries — the same
overlapping chunk strategy from Chapter 10.

**`FAISS.from_documents()`**
Embeds all chunks and builds the FAISS index in a single call. Replaces the
manual encode loop + `index.add()` from the scratch version.

**`vectorstore.save_local("faiss_index")`**
Saves two files to the `faiss_index/` folder:
- `index.faiss` — the vector index
- `index.pkl` — the document metadata

---

### app.py — Phase B: The Query Pipeline

**Runs on every question.** Loads the saved index, builds the LCEL chain,
manages conversation memory, and streams responses.

#### The LCEL chain

```python
rag_chain = (
    {
        "context":      RunnableLambda(get_question) | retriever | format_docs,
        "question":     RunnableLambda(get_question),
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Data flows left to right through the pipe `|` operator:

1. `RunnableLambda(get_question)` — extracts the question string from the
   input dict (required because `RunnableWithMessageHistory` passes a dict)
2. `retriever` — searches FAISS for the 2 most relevant policy chunks
3. `format_docs` — formats retrieved chunks into a readable context string
4. `prompt` — fills `{context}`, `{question}`, and `{chat_history}` into
   the `ChatPromptTemplate`
5. `llm` — sends the filled prompt to Claude
6. `StrOutputParser()` — extracts the plain text from Claude's response

#### Memory — RunnableWithMessageHistory

Modern LangChain (v0.3+) uses `RunnableWithMessageHistory` instead of the
legacy `ConversationBufferMemory`. It wraps the chain and automatically:
- Injects `chat_history` into the input dict before each call
- Saves the new exchange to history after each call
- Keeps separate history per `session_id`

```python
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)
```

`store` is a dict mapping `session_id` → `ChatMessageHistory`. Each user
or conversation gets its own isolated history.

**Important:** `store` lives in RAM. It resets when the process restarts.
For persistent memory across sessions, replace `ChatMessageHistory` with
`RedisChatMessageHistory` or `DynamoDBChatMessageHistory`.

#### Streaming

Any LCEL chain supports streaming with one method change:

```python
# Non-streaming
answer = chain_with_history.invoke(input_dict, config=config)

# Streaming — tokens appear word by word
for chunk in chain_with_history.stream(input_dict, config=config):
    print(chunk, end="", flush=True)
```

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
Created 5 chunks from 5 documents
Index saved to faiss_index/ folder
Total vectors: 5
```

This creates the `faiss_index/` folder. Only re-run when your documents change.

### Step 2 — Run the RAG pipeline

```bash
python app.py
```

Expected output (streamed word by word):
```
Q: My package arrived broken. What do I do?
A: I'm sorry to hear that! Please contact us at support@store.com
   within 48 hours with a photo of the damage. We'll send a
   replacement at no charge within 3 business days. [doc_3]

Q: How quickly will the replacement arrive?
A: As mentioned, your replacement will be shipped within 3 business
   days at no charge. [doc_3]

Q: Can I cancel my order from 3 hours ago?
A: Unfortunately, orders can only be cancelled within 1 hour of
   placement. Since 3 hours have passed, your order is now in
   processing and cannot be cancelled. [doc_4]
```

Notice the second question — "How quickly will the replacement arrive?" —
references "the replacement" from the previous turn. Memory is working.

---

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain.text_splitter` | Old import path | `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| `ModuleNotFoundError: langchain.memory` | Legacy API removed | Use `RunnableWithMessageHistory` (see app.py) |
| `AttributeError: 'AddableDict' has no 'replace'` | Retriever receiving dict not string | Use `RunnableLambda(get_question)` before retriever |
| `KeyError: chat_history missing` | Chain not passing history to prompt | Add `"chat_history": RunnableLambda(lambda x: x.get(...))` to chain dict |
| `allow_dangerous_deserialization` error | Loading local FAISS index | Pass `allow_dangerous_deserialization=True` to `load_local()` |
| `No such file: faiss_index` | indexer.py not run yet | Run `python indexer.py` first |

---

## LangChain version note

This project uses **LangChain v0.3+**. LangChain has changed its import
paths across versions. If you see import errors, check your version:

```bash
pip show langchain
```

Common import path changes in v0.3:
- `langchain.text_splitter` → `langchain_text_splitters`
- `langchain.memory` → `langchain_community.chat_message_histories`
- `langchain.embeddings` → `langchain_community.embeddings`
- `langchain.vectorstores` → `langchain_community.vectorstores`

---

## Key concepts this code teaches

| Concept | Where to look |
|---|---|
| LCEL pipe operator | `rag_chain` in `app.py` |
| RecursiveCharacterTextSplitter | `indexer.py` — chunk_size and overlap |
| Retriever from vectorstore | `vectorstore.as_retriever()` in `app.py` |
| RunnableLambda for dict extraction | `get_question` function in `app.py` |
| Modern memory pattern | `RunnableWithMessageHistory` in `app.py` |
| Session-based history | `store` dict and `get_session_history()` |
| Streaming in LCEL | `.stream()` in `chat()` function |

---

## Dependencies explained

```
langchain>=0.3.0              # LangChain core framework
langchain-core>=0.3.0         # Core abstractions (runnables, prompts)
langchain-anthropic>=0.3.0    # Claude integration for LangChain
langchain-community>=0.3.0    # FAISS, HuggingFace, chat histories
langchain-text-splitters>=0.3.0  # RecursiveCharacterTextSplitter
faiss-cpu>=1.8.0              # Vector similarity search
sentence-transformers>=3.0.0  # Local embedding model
numpy>=1.26.0                 # Array operations
anthropic>=0.40.0             # Anthropic SDK (used in non-LC examples)
```

---

## Author

Built as part of the *"From Tokens to Agents"* AI learning series.
Part 3 — Building LLM Applications · Chapter 26.
