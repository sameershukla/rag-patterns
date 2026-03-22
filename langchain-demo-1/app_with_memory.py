from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app import retriever, format_docs, llm

# ── Memory object ─────────────────────────────────────────────
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ── Updated prompt — includes chat history ────────────────────
prompt_with_memory = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful customer support agent. "
        "Answer using ONLY the store policies in the context. "
        "Be friendly and concise."
    ),
    MessagesPlaceholder(variable_name="chat_history"),  # ← history slot
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
])

# ── RAG chain with memory ─────────────────────────────────────
rag_chain_with_memory = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
    }
    | prompt_with_memory
    | llm
    | StrOutputParser()
)

# ── Chat function — saves history after each turn ─────────────
def chat(question: str) -> str:
    answer = rag_chain_with_memory.invoke(question)

    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    return answer

# ── Test conversational memory ────────────────────────────────
print(chat("My package arrived broken. What do I do?"))
print(chat("How quickly will the replacement arrive?"))

# ↑ "The replacement" refers to context from the previous turn
# Without memory → "I'm not sure what replacement you mean"
# With memory → "Your replacement will ship within 3 business days"
