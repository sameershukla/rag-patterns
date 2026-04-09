import os

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY before running this script.")

    # 1) Sample knowledge base
    documents = [
        Document(
            page_content=(
                "LangChain is a framework for building applications with language models. "
                "It helps connect prompts, models, tools, retrievers, and output parsers."
            )
        ),
        Document(
            page_content=(
                "RAG stands for Retrieval-Augmented Generation. "
                "In RAG, the system retrieves relevant documents first and then uses them as context for the model."
            )
        ),
        Document(
            page_content=(
                "A retriever accepts a query and returns relevant documents. "
                "This helps ground answers in external knowledge instead of relying only on model memory."
            )
        ),
        Document(
            page_content=(
                "Vector stores keep embedded document chunks and support similarity search. "
                "This allows the system to find text that is semantically related to the user query."
            )
        ),
    ]

    # 2) Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )
    chunks = splitter.split_documents(documents)

    # 3) Create embeddings locally
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 4) Store chunks in an in-memory vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # 5) Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 6) Create LLM
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
    )

    # 7) Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise assistant. "
                "Answer only from the retrieved context. "
                "If the answer is not present in the context, say: "
                "'The answer is not available in the retrieved context.'"
            ),
            (
                "human",
                "Retrieved context:\n{context}\n\nQuestion:\n{question}"
            ),
        ]
    )

    parser = StrOutputParser()

    question = "What is RAG and why is a retriever important?"

    # 8) Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)

    # 9) Generate grounded answer
    chain = prompt | model | parser
    result = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print("Question:\n")
    print(question)

    print("\nRetrieved Context:\n")
    print(context)

    print("\nAnswer:\n")
    print(result)


if __name__ == "__main__":
    main()
