"""
app.py — Phase B: Querying with Phase 2 hallucination controls

Applies all four advanced grounding techniques:
  1. Explicit "I don't know" instruction
  2. Citation requirement — LLM must cite which doc each claim comes from
  3. Confidence scoring — LLM rates its own certainty (1-5)
  4. Context-first prompt structure — context before question

The answer is only returned if confidence >= 3.
Below that, a safe fallback is returned instead.
"""

import anthropic

from Retriever import Retriever

# ── Setup ─────────────────────────────────────────────────────────────────────

retriever = Retriever()
client    = anthropic.Anthropic()

# ── Confidence threshold ──────────────────────────────────────────────────────
# LLM rates its answer 1-5. Below this, return a safe fallback.
# 1 = not in context at all
# 2 = loosely related
# 3 = partially supported
# 4 = mostly supported
# 5 = fully and directly supported
MIN_CONFIDENCE = 3


# ── Parse the structured response ────────────────────────────────────────────

def parse_response(raw: str) -> tuple[int, str]:
    """
    Parse the structured LLM output into (confidence_score, answer).

    Expected format:
        CONFIDENCE: 4
        ANSWER: The maximum daily dose is 4000mg...
    """
    confidence = 1
    answer     = raw.strip()

    lines = raw.strip().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = int(line.split(":")[1].strip())
            except ValueError:
                confidence = 1
        elif line.startswith("ANSWER:"):
            # Everything from this line onwards is the answer
            answer = "\n".join(lines[i:]).replace("ANSWER:", "", 1).strip()
            break

    return confidence, answer


# ── Core query function ───────────────────────────────────────────────────────

def answer_question(question: str) -> dict:
    """
    Full RAG query pipeline with Phase 2 hallucination controls.

    Returns a dict with:
        - answer:     the answer string (or fallback message)
        - confidence: the LLM's self-rated confidence (1-5)
        - sources:    list of source doc IDs used
        - fallback:   True if a fallback was returned instead of an answer
    """

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    results = retriever.retrieve(question)

    if results is None:
        return {
            "answer":     "I don't have information about that in our clinic guidelines. "
                          "Please speak directly with one of our medical staff.",
            "confidence": 0,
            "sources":    [],
            "fallback":   True,
        }

    # ── Step 2: Build context with document labels ────────────────────────────
    # Labels like [Doc 1], [Doc 2] are what the LLM uses for citations.
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Doc {i+1}] (Source: {r['source']} | Relevance: {r['score']})\n{r['text']}"
        )
    context = "\n\n".join(context_parts)

    # ── Step 3: Build the augmented prompt ───────────────────────────────────
    # Technique 1: Context-first structure (context before question)
    # Technique 2: Explicit "I don't know" instruction
    # Technique 3: Citation requirement
    # Technique 4: Confidence scoring
    prompt = f"""You are a medical clinic assistant. Your answers must be precise and safe.

=== CLINIC GUIDELINES ===
{context}

=== PATIENT QUESTION ===
{question}

=== INSTRUCTIONS ===
Answer using ONLY the clinic guidelines above.
- After every factual claim, cite the source using [Doc 1] or [Doc 2].
- If the answer is NOT explicitly in the guidelines, do not guess.
  Instead write: "This is not covered in our current guidelines. Please speak with a medical professional."
- Never estimate, infer, or use outside medical knowledge.

Respond in this exact format:
CONFIDENCE: [1-5 where 5=fully supported by the docs above, 1=not in the docs at all]
ANSWER: [your cited answer]"""

    # ── Step 4: Generate with Claude ─────────────────────────────────────────
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text

    # ── Step 5: Parse structured output ──────────────────────────────────────
    confidence, answer = parse_response(raw)

    # ── Step 6: Confidence gate ───────────────────────────────────────────────
    # If the LLM itself is not confident, don't trust the answer.
    if confidence < MIN_CONFIDENCE:
        return {
            "answer":     "I'm not confident enough to answer this from our current guidelines. "
                          "Please speak directly with a medical professional.",
            "confidence": confidence,
            "sources":    [r["source"] for r in results],
            "fallback":   True,
        }

    return {
        "answer":     answer,
        "confidence": confidence,
        "sources":    [r["source"] for r in results],
        "fallback":   False,
    }


# ── Pretty print helper ───────────────────────────────────────────────────────

def print_result(question: str, result: dict):
    confidence_bar = "█" * result["confidence"] + "░" * (5 - result["confidence"])
    fallback_flag  = " [FALLBACK]" if result["fallback"] else ""

    print(f"\n{'─' * 65}")
    print(f"Q: {question}")
    print(f"Confidence: [{confidence_bar}] {result['confidence']}/5{fallback_flag}")
    print(f"Sources: {', '.join(result['sources']) if result['sources'] else 'none'}")
    print(f"\nA: {result['answer']}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "How much paracetamol can I take in a day?",
        "How do I care for my wound after surgery?",
        "How long do I need to fast before a blood test?",
        "Can I give aspirin to my 10-year-old?",
        "What should I do if I miss a dose of antibiotics?",  # Not in the docs — tests fallback
        "What is the weather forecast for tomorrow?",          # Completely off-topic — tests fallback
    ]

    print("RAG HALLUCINATION ADVANCED DEMO")
    print("Grounding techniques: citations + confidence scoring + context-first + explicit fallback\n")

    for q in questions:
        result = answer_question(q)
        print_result(q, result)

    print(f"\n{'─' * 65}")
    print("All logs saved to rag_logs.jsonl")
    print("Run eval.py to measure your accuracy rate.")
