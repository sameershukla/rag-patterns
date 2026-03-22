from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Documents — same knowledge base as Chapter 13 ────────────
raw_docs = [
    {
        "id": "doc_1",
        "text": (
            "Return Policy: Customers can return any item "
            "within 30 days. Items must be unused and in "
            "original packaging. Refunds within 5-7 days."
        )
    },
    {
        "id": "doc_2",
        "text": (
            "Shipping: Standard 5-7 days $4.99. Express 2-3 "
            "days $12.99. Free shipping on orders over $50."
        )
    },
    {
        "id": "doc_3",
        "text": (
            "Damaged Items: Contact within 48 hours. Send a "
            "photo to support@store.com. Replacement shipped "
            "at no charge within 3 business days."
        )
    },
    {
        "id": "doc_4",
        "text": (
            "Order Cancellation: Cancel within 1 hour of "
            "placement. After 1 hour order is in processing "
            "and cannot be cancelled."
        )
    },
    {
        "id": "doc_5",
        "text": (
            "Gift Cards: Never expire, cannot be exchanged "
            "for cash. Lost cards not replaced without receipt."
        )
    },
]

# ── Wrap in LangChain Document objects ────────────────────────
# Document = text + metadata. Metadata travels with the chunk.
documents = [
    Document(
        page_content=d["text"],
        metadata={"source": d["id"]}
    )
    for d in raw_docs
]

# ── Chunk the documents ───────────────────────────────────────
# RecursiveCharacterTextSplitter is the production standard.
# It tries to split on paragraphs, then sentences, then words.
# chunk_overlap creates overlapping windows (Chapter 10 concept)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # max characters per chunk
    chunk_overlap=30,   # overlap between consecutive chunks
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")

# ── Embed and build FAISS index ───────────────────────────────
# Same model as your scratch version — all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# from_documents: embeds all chunks and builds the index in one call
vectorstore = FAISS.from_documents(chunks, embeddings)

# ── Save to disk ──────────────────────────────────────────────
vectorstore.save_local("faiss_index")

print("Index saved to faiss_index/ folder")
print(f"Total vectors: {vectorstore.index.ntotal}")
