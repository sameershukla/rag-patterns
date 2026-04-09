import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY before running this script.")

    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise assistant. "
                "Answer only from the provided context. "
                "If the answer is not present in the context, say: "
                "'The answer is not available in the provided context.'"
            ),
            (
                "user",
                "Context:\n{context}\n\nQuestion:\n{question}"
            ),
        ]
    )

    parser = StrOutputParser()
    chain = prompt | model | parser
    context = """
        LangChain is a framework for building applications powered by language models.
        It helps developers compose prompts, models, output parsers, retrievers, tools, and workflows.
        Output parsers are used to convert model responses into cleaner formats such as strings, JSON, or structured objects.
        LCEL stands for LangChain Expression Language and uses the pipe operator to connect components.
    """
    question = "What is LCEL and what is it used for?"
    result = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )
    print("Question:\n")
    print(question)
    print("\nAnswer:\n")
    print(result)

if __name__ == "__main__":
    main()
