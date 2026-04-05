from typing import List


# -----------------------------------
# Sample Data (Simulated RAG Output)
# -----------------------------------
query = "What is the maximum DPU limit for an AWS Glue job?"

generated_answer = "The maximum DPU limit for an AWS Glue job is 100."

retrieved_docs = [
    "AWS Glue allows a maximum of 100 DPUs per job.",
    "Glue pricing depends on DPU usage.",
    "AWS Lambda has concurrency limits."
]

ground_truth_answer = "The maximum DPU limit for an AWS Glue job is 100."


# -----------------------------------
# Helper: simple text match
# -----------------------------------
def contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


# -----------------------------------
# 1. Faithfulness
# -----------------------------------
def faithfulness(answer: str, context_docs: List[str]) -> float:
    """
    Check if answer is supported by retrieved documents.
    """
    for doc in context_docs:
        if contains(doc, answer):
            return 1.0
    return 0.0


# -----------------------------------
# 2. Answer Relevance
# -----------------------------------
def answer_relevance(query: str, answer: str) -> float:
    """
    Check if answer addresses the query.
    """
    key_terms = query.lower().split()

    matches = sum(1 for term in key_terms if term in answer.lower())
    return matches / len(key_terms)


# -----------------------------------
# 3. Context Precision
# -----------------------------------
def context_precision(retrieved_docs: List[str], ground_truth: str) -> float:
    """
    Fraction of retrieved docs that are actually relevant.
    """
    relevant_docs = 0

    for doc in retrieved_docs:
        if contains(doc, ground_truth):
            relevant_docs += 1

    return relevant_docs / len(retrieved_docs)


# -----------------------------------
# 4. Context Recall
# -----------------------------------
def context_recall(retrieved_docs: List[str], ground_truth: str) -> float:
    """
    Did we retrieve the necessary information?
    """
    for doc in retrieved_docs:
        if contains(doc, ground_truth):
            return 1.0
    return 0.0


# -----------------------------------
# Run Evaluation
# -----------------------------------
if __name__ == "__main__":
    f = faithfulness(generated_answer, retrieved_docs)
    r = answer_relevance(query, generated_answer)
    p = context_precision(retrieved_docs, ground_truth_answer)
    rc = context_recall(retrieved_docs, ground_truth_answer)

    print("\nRAG Evaluation Metrics:\n")
    print(f"Faithfulness       : {f:.2f}")
    print(f"Answer Relevance   : {r:.2f}")
    print(f"Context Precision  : {p:.2f}")
    print(f"Context Recall     : {rc:.2f}")
