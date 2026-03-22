"""
app.py — Complete LangChain RAG pipeline
Modern LangChain (v0.3+) memory using RunnableWithMessageHistory
"""

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ── Load the saved index ──────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ── Create retriever ──────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# ── Define the LLM ────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=300
)

# ── Prompt template with chat history slot ────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful customer support agent for an e-commerce store. "
     "Answer using ONLY the store policies provided in the context. "
     "If the answer is not in the context, say you don't have that information. "
     "Be friendly and concise."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# ── Format retrieved documents into a context string ──────────
def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
        for doc in docs
    )

def get_question(x):
    return x["question"]

# ── Core RAG chain (no memory yet) ───────────────────────────
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
# ── Memory store — one ChatMessageHistory per session ─────────
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ── Wrap chain with message history ───────────────────────────
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# ── Chat function ─────────────────────────────────────────────
def chat(question: str, session_id: str = "default", stream: bool = False) -> str:
    config = {"configurable": {"session_id": session_id}}
    input_dict = {"question": question}

    if stream:
        answer = ""
        for chunk in chain_with_history.stream(input_dict, config=config):
            print(chunk, end="", flush=True)
            answer += chunk
        print()
        return answer
    else:
        return chain_with_history.invoke(input_dict, config=config)


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Q: My package arrived broken. What do I do?")
    print(f"A: {chat('My package arrived broken. What do I do?', stream=True)}\n")

    print("Q: How quickly will the replacement arrive?")
    print(f"A: {chat('How quickly will the replacement arrive?', stream=True)}\n")

    print("Q: Can I cancel my order from 3 hours ago?")
    print(f"A: {chat('Can I cancel my order from 3 hours ago?', stream=True)}\n")
