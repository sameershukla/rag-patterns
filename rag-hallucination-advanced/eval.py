"""
eval.py — Golden Evaluation Set
Runs 10 known questions against the RAG system and measures accuracy.

This is how you know if a change made things better or worse.
Run this after every change to chunk size, threshold, prompt, or model.

Scoring:
  - Each question has a list of terms that MUST appear in the answer
  - If all must_contain terms are present → PASS
  - If any are missing → FAIL
  - Final score = (passed / total) * 100
"""
from App import answer_question

# ── Golden evaluation set ─────────────────────────────────────────────────────
# 10 questions with known correct answers derived from the clinic documents.
# must_contain: all of these strings must appear in the answer (case-insensitive)
# should_fallback: True if this question has no answer in the docs

eval_set = [
    {
        "id":             "q01",
        "question":       "What is the maximum daily dose of paracetamol for adults?",
        "must_contain":   ["4000", "4g"],
        "should_fallback": False,
        "note":           "Exact number from doc_1",
    },
    {
        "id":             "q02",
        "question":       "How long must I keep my wound dry after surgery?",
        "must_contain":   ["48"],
        "should_fallback": False,
        "note":           "48 hours from doc_2",
    },
    {
        "id":             "q03",
        "question":       "What is the cancellation fee if I cancel less than 24 hours before?",
        "must_contain":   ["50"],
        "should_fallback": False,
        "note":           "$50 fee from doc_3",
    },
    {
        "id":             "q04",
        "question":       "How long do I need to fast before a blood test?",
        "must_contain":   ["8", "12"],
        "should_fallback": False,
        "note":           "8 to 12 hours from doc_5",
    },
    {
        "id":             "q05",
        "question":       "Can I give aspirin to my 8-year-old child?",
        "must_contain":   ["16"],
        "should_fallback": False,
        "note":           "No aspirin under 16 from doc_6",
    },
    {
        "id":             "q06",
        "question":       "How many times a day is Amoxicillin prescribed?",
        "must_contain":   ["three", "3"],
        "should_fallback": False,
        "note":           "Three times daily from doc_4",
    },
    {
        "id":             "q07",
        "question":       "At what fever temperature should I contact the clinic after surgery?",
        "must_contain":   ["38.5"],
        "should_fallback": False,
        "note":           "38.5 degrees from doc_2",
    },
    {
        "id":             "q08",
        "question":       "What days are blood tests available?",
        "must_contain":   ["monday", "friday"],
        "should_fallback": False,
        "note":           "Monday to Friday from doc_5",
    },
    {
        "id":             "q09",
        "question":       "What is the paracetamol limit for patients with liver disease?",
        "must_contain":   ["2000"],
        "should_fallback": False,
        "note":           "2000mg limit from doc_1",
    },
    {
        "id":             "q10",
        "question":       "What is the best treatment for a broken leg?",
        "must_contain":   [],
        "should_fallback": True,
        "note":           "Not in docs — system must return a fallback",
    },
]


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_eval():
    passed  = 0
    failed  = 0
    results = []

    print("=" * 65)
    print("RUNNING GOLDEN EVALUATION SET")
    print(f"Total questions: {len(eval_set)}")
    print("=" * 65)

    for item in eval_set:
        result = answer_question(item["question"])
        answer = result["answer"].lower()

        # ── Check fallback questions ──────────────────────────────────────────
        if item["should_fallback"]:
            # Question has no answer in docs — system should return fallback
            if result["fallback"]:
                status = "PASS"
                passed += 1
                reason = "correctly returned fallback"
            else:
                status = "FAIL"
                failed += 1
                reason = "should have returned fallback but gave an answer"

        # ── Check answerable questions ────────────────────────────────────────
        else:
            # All must_contain terms must appear in the answer
            missing = [
                term for term in item["must_contain"]
                if term.lower() not in answer
            ]
            if not missing:
                status = "PASS"
                passed += 1
                reason = "all required terms found"
            else:
                status = "FAIL"
                failed += 1
                reason = f"missing terms: {missing}"

        results.append({
            "id":         item["id"],
            "status":     status,
            "confidence": result["confidence"],
            "fallback":   result["fallback"],
            "reason":     reason,
            "question":   item["question"],
        })

        icon = "✓" if status == "PASS" else "✗"
        print(f"{icon} [{item['id']}] conf={result['confidence']}/5 | {reason}")
        print(f"   Q: {item['question'][:70]}")
        if status == "FAIL":
            print(f"   A: {result['answer'][:120]}...")
        print()

    # ── Final score ───────────────────────────────────────────────────────────
    score = (passed / len(eval_set)) * 100

    print("=" * 65)
    print(f"EVAL COMPLETE")
    print(f"Score:  {score:.0f}%  ({passed} passed / {failed} failed out of {len(eval_set)})")
    print("=" * 65)

    if score >= 90:
        print("Production ready — score above 90%")
    elif score >= 70:
        print("Needs improvement before production — tune retrieval and prompts")
    else:
        print("Not production ready — significant issues in retrieval or grounding")

    return score


if __name__ == "__main__":
    run_eval()
