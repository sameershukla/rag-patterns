import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

def build_chain():
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=300
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an helpful assistant that explains concepts simply"),
        ("user", "Explain {topic} in {style}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain

def explain_topic(chain, topic: str, style: str) -> str:
    return chain.invoke({
        "topic": topic,
        "style": style
    })


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY before running this script.")
    chain = build_chain()
    result_1 = explain_topic(
        chain,
        topic = "What LangChain is",
        style = "3 short beginner-friendly lines"
    )
    print(result_1)
    print ('-----------------------------')
    result_2 = explain_topic(
        chain,
        topic = "Is it going to rain in Dallas on coming Sunday",
        style = "Answer in %"
    )
    print(result_2)

if __name__ == "__main__":
    main()
