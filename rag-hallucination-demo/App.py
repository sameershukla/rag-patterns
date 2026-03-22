"""
app.py — Phase B: Querying
Runs each patient question TWICE:
  1. WITHOUT RAG — raw LLM answer (may hallucinate)
  2. WITH RAG    — grounded answer from clinic documents

The side-by-side output makes the hallucination problem visible.
"""

import anthropic

from Retriever import Retriever

# ── Setup ─────────────────────────────────────────────────────────────────────

retriever = Retriever()
client    = anthropic.Anthropic()

# ── Without RAG — raw LLM, no retrieved context ───────────────────────────────

def answer_without_rag(question: str) -> str:
    """
    Send the question directly to Claude with no retrieved context.
    The LLM answers purely from its frozen training weights.
    This is where hallucination happens.
    """
    prompt = f"""You are a medical clinic assistant.
Answer the patient's question as helpfully as you can.

Patient question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ── With RAG — grounded answer from clinic documents ──────────────────────────

def answer_with_rag(question: str) -> str:
    """
    Retrieve relevant clinic documents first, then send them
    alongside the question to Claude. The LLM answers only from
    what we retrieved — no guessing, no hallucination.
    """
    # Step 1: Retrieve relevant documents
    results = retriever.retrieve(question)

    # Step 2: If no relevant document found, return safe fallback
    if results is None:
        return (
            "I don't have specific clinic guidelines for that question. "
            "Please speak directly with one of our medical staff."
        )

    # Step 3: Build context from retrieved documents
    context = "\n\n".join([
        f"[Clinic Document {i+1} | Relevance: {r['score']}]\n{r['text']}"
        for i, r in enumerate(results)
    ])

    # Step 4: Build the augmented prompt
    prompt = f"""You are a medical clinic assistant.
Answer the patient's question using ONLY the clinic guidelines provided below.
Be clear and precise — this is a medical context where accuracy is critical.
If the answer is not in the guidelines, say you don't have that information
and advise the patient to speak with a medical professional.
Never guess or estimate medical information.

=== CLINIC GUIDELINES ===
{context}

=== PATIENT QUESTION ===
{question}

=== YOUR ANSWER ==="""

    # Step 5: Generate grounded answer with Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ── Run comparison ────────────────────────────────────────────────────────────

def run_comparison(question: str):
    """Run a single question through both pipelines and print the comparison."""
    print(f"\n{'=' * 65}")
    print(f"PATIENT QUESTION: {question}")
    print(f"{'=' * 65}")

    print("\n[ WITHOUT RAG — raw LLM, no clinic context ]")
    print("-" * 65)
    print(answer_without_rag(question))

    print("\n[ WITH RAG — grounded from clinic documents ]")
    print("-" * 65)
    print(answer_with_rag(question))
    print()


if __name__ == "__main__":

    questions = [
        "How much paracetamol can I take in a day?",
        "How do I care for my wound after surgery?",
        "How long do I need to fast before a blood test?",
        "What is the cancellation fee if I miss my appointment?",
        "My baby has a fever. What should I do?",
    ]

    print("\nRAG HALLUCINATION DEMO — Medical Clinic FAQ Bot")
    print("Each question is answered twice:")
    print("  1. WITHOUT RAG — raw LLM (may hallucinate specific details)")
    print("  2. WITH RAG    — grounded from actual clinic documents\n")

    for question in questions:
        run_comparison(question)
