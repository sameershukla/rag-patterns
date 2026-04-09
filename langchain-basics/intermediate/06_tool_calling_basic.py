import os
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Step 1
@tool
def multiply(a, b):
    """Multiply two numbers and return the result."""
    return a * b


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set environment variable ANTHROPIC_API_KEY")

    #Step 2 create a model
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=300
    ).bind_tools([multiply])

    # Step 3: Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "You MUST use tools for any mathematical calculation. "
                "Do not answer directly."
            ),
            ("user", "{question}"),
        ]
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser
    question = "What is 7 multiplied by 8?"
    result = chain.invoke({"question": question})
    print("Question:\n")
    print(question)
    print("\nAnswer:\n")
    print(result)

if __name__ == "__main__":
    main()
