# LangChain Intermediate: From Structured Output to Tool Usage

This module builds on the basics and introduces real-world concepts required to move toward production-grade LLM applications.

In these examples, we go beyond simple text generation and start focusing on:
- structured outputs
- grounding responses using context
- enabling models to use external tools

---

## 📂 Project Structure

```text
langchain-basics/
└── intermediate/
    ├── 04_structured_output.py
    ├── 05_grounded_answer_from_context.py
    ├── 06_tool_calling_basic.py


# Exercise 04: Structured Output

In basic examples, the model returns free-form text.
This is hard to use in real applications where structured data is required.

Example need:

topic
summary
difficulty
key points

#Solution: Use Pydantic schema + structured output

$ Exercise 05: Grounded Answer from Context
Models answer from general knowledge, which may:

be irrelevant
hallucinate
not follow your data

#Solution: Provide explicit context and force the model to answer from it.

$ Exercise 06: Tool Calling (Concept Introduction)
Models can generate text, but cannot:

perform real calculations reliably
call APIs
execute logic

#Solution: Bind tools (Python functions) to the model.
