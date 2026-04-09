import os
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

class ConcreteExample(BaseModel):
    topic: str = Field(description="The topic was explained")
    summary: str = Field(description="A short beginner-friendly explanation")
    difficulty: str = Field(description="Difficulty level such as beginner, intermediate, or advanced")
    key_points: List[str] = Field(description="Important points about the topic")

def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY before running this script.")

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=300
    )

    structured_model = llm.with_structured_output(ConcreteExample)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI tutor. "
                "Explain technical concepts clearly for beginners. "
                "Always provide concise, accurate, and structured responses."
            ),
            (
                "user",
                "Explain the topic: {topic}"
            ),
        ]
    )
    chain = prompt | structured_model
    result = chain.invoke({"topic": "LangChain output parsers"})
    print("Structured response:\n")
    print(f"Topic      : {result.topic}")
    print(f"Summary    : {result.summary}")
    print(f"Difficulty : {result.difficulty}")
    print("Key Points :")
    for idx, point in enumerate(result.key_points, start=1):
        print(f"  {idx}. {point}")

if __name__ == "__main__":
    main()
