"""
app.py — Phase B: Querying
Loads the retriever and handles customer questions end-to-end.
This is the live, user-facing pipeline. Runs on every question.
"""

import anthropic

from Retriever import Retriever

# ── Setup ─────────────────────────────────────────────────────────────────────

retriever = Retriever()
client = anthropic.Anthropic()


# ── Query Pipeline ────────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    """
    Full RAG query pipeline:
    1. Retrieve relevant chunks from the index
    2. Build augmented prompt
    3. Generate answer with Claude
    """

    # Step 1: Retrieve
    results = retriever.retrieve(question)

    if results is None:
        return (
            "I don't have information about that in our current policies. "
            "Please contact support@store.com for further help."
        )

    # Step 2: Build context from retrieved docs
    context = "\n\n".join([
        f"[Policy {i + 1} | Relevance: {r['score']}]\n{r['text']}"
        for i, r in enumerate(results)
    ])

    # Step 3: Augment the prompt
    prompt = f"""You are a helpful customer support agent for an e-commerce store.
Answer the customer's question using ONLY the policy information provided below.
Be friendly, clear, and concise.
If the answer is not in the policies, say you don't have that information.

=== STORE POLICIES ===
{context}

=== CUSTOMER QUESTION ===
{question}

=== YOUR ANSWER ==="""

    # Step 4: Generate with Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "My package arrived broken. What do I do?",
        "How long does free shipping take?",
        "Can I cancel my order from 3 hours ago?",
    ]

    for question in test_questions:
        print(f"Customer: {question}")
        answer = answer_question(question)
        print(f"Bot:      {answer}")
        print("-" * 60)
