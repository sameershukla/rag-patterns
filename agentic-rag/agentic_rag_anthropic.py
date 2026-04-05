import json
import os
from typing import Any, Dict, List

from anthropic import Anthropic

# --------------------------------------------------
# Config
# --------------------------------------------------
MAX_ITERATIONS = 5
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# --------------------------------------------------
# Mock knowledge base
# --------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "title": "AWS Glue Standard execution class",
        "content": "AWS Glue Standard execution class is optimized for faster startup and general ETL workloads."
    },
    {
        "title": "AWS Glue Flex execution class",
        "content": "AWS Glue Flex execution class is lower cost but startup can be delayed. It is suitable for non-urgent jobs."
    },
    {
        "title": "AWS Glue execution class comparison",
        "content": "Standard is better for predictable latency. Flex is better for cost-sensitive workloads that can tolerate delays."
    },
    {
        "title": "Nightly ETL guidance",
        "content": "Nightly ETL jobs usually prefer predictable completion time over startup delay, especially when downstream dependencies exist."
    },
]

# --------------------------------------------------
# Tool: search knowledge base
# --------------------------------------------------
def search_knowledge_base(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Very simple keyword search for demonstration.
    In production, replace this with vector search or hybrid search.
    """
    query_terms = set(query.lower().split())
    results: List[Dict[str, Any]] = []

    for doc in KNOWLEDGE_BASE:
        text = f"{doc['title']} {doc['content']}".lower()
        score = sum(1 for term in query_terms if term in text)
        if score > 0:
            results.append(
                {
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": score,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the internal knowledge base for relevant documents. "
            "Use this when you need factual information before answering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A refined search query written by the model."
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many results to return.",
                    "default": 3
                },
            },
            "required": ["query"],
        },
    }
]

SYSTEM_PROMPT = """
You are an assistant using agentic RAG.

You have access to a tool called search_knowledge_base.
Use it when the question needs external information.
You may search multiple times if needed.
Write your own refined search queries instead of blindly copying the user's wording.

When you have enough information, answer clearly and briefly.
If the tool results are insufficient, you may search again.
Do not loop forever. Prefer to answer once enough evidence is available.
"""


def extract_text_from_response_blocks(blocks: List[Any]) -> str:
    parts: List[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def has_tool_use(blocks: List[Any]) -> bool:
    return any(getattr(block, "type", None) == "tool_use" for block in blocks)


def run_agentic_rag(user_question: str) -> str:
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": user_question,
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Print assistant text if present
        assistant_text = extract_text_from_response_blocks(response.content)
        if assistant_text:
            print(f"\n[Iteration {iteration}] Assistant thinking/output:")
            print(assistant_text)

        # If no tool call, Claude has produced a final answer
        if not has_tool_use(response.content):
            return assistant_text or "No final answer returned."

        # Add assistant tool_use message to conversation exactly as returned
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

        # Execute all tool calls in this turn
        tool_result_blocks = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "search_knowledge_base":
                query = block.input["query"]
                top_k = block.input.get("top_k", 3)

                print(f"\n[Iteration {iteration}] Tool call:")
                print(f"search_knowledge_base(query={query!r}, top_k={top_k})")

                results = search_knowledge_base(query=query, top_k=top_k)

                print("[Tool results]")
                for idx, result in enumerate(results, start=1):
                    print(f"{idx}. {result['title']} (score={result['score']})")

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(results),
                    }
                )

        # Return tool results to Claude as the next user message
        messages.append(
            {
                "role": "user",
                "content": tool_result_blocks,
            }
        )

    return "Reached MAX_ITERATIONS without a final answer. Return the best partial answer or tighten the prompt."


if __name__ == "__main__":
    question = (
        "Compare AWS Glue Standard vs Flex execution class and tell me which is "
        "better for our nightly ETL job."
    )
    final_answer = run_agentic_rag(question)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(final_answer)
