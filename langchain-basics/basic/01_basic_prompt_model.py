import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY before running this script.")

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=300
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an helpful assistant that explains concepts simply"),
        ("user", "Explain {topic} in 3 simple lines")
    ])

    chain = prompt | llm
    response = chain.invoke({"topic": "What LangChain is and put numbering before every topic?"})
    print(response.content)

if __name__ == "__main__":
    main()
