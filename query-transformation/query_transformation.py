from typing import List, Dict


# -----------------------------------
# 1. Query Expansion
# -----------------------------------
SYNONYM_MAP: Dict[str, List[str]] = {
    "oom": ["out of memory", "memory error"],
    "glue": ["aws glue", "glue job"],
    "retry": ["retries", "retry strategy", "retry mechanism"],
    "failure": ["error", "issue", "job failure", "pipeline failure"],
    "timeout": ["timed out", "request timeout"],
}


def expand_query(query: str, synonym_map: Dict[str, List[str]]) -> str:
    """
    Expand the query by appending related terms for important keywords.
    """
    terms = query.lower().split()
    expanded_terms = list(terms)

    for term in terms:
        if term in synonym_map:
            expanded_terms.extend(synonym_map[term])

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)

    return " ".join(unique_terms)


# -----------------------------------
# 2. Multi-Query Generation
# -----------------------------------
def generate_multi_queries(query: str) -> List[str]:
    """
    Create multiple alternative versions of the same user query.
    In production, this is often done with an LLM.
    Here we simulate it with simple templates.
    """
    query = query.strip()

    return [
        query,
        f"What are the common causes of {query}?",
        f"How do I troubleshoot {query}?",
        f"Best way to fix {query}",
    ]


# -----------------------------------
# 3. Query Rewriting
# -----------------------------------
def rewrite_query(query: str) -> str:
    """
    Rewrite a vague user query into a clearer search-oriented query.
    This is rule-based for demonstration purposes.
    In production, an LLM often does this step.
    """
    lowered = query.lower().strip()

    rewrite_rules = {
        "why glue fails": "AWS Glue job failure reasons, out of memory errors, configuration issues, IAM permission problems",
        "glue oom": "AWS Glue out of memory error causes, memory tuning, partition sizing, Spark executor memory issues",
        "retry issue": "retry strategy problems, backoff retry patterns, transient failure handling",
    }

    return rewrite_rules.get(lowered, query)


# -----------------------------------
# Demo
# -----------------------------------
if __name__ == "__main__":
    user_query = "glue oom"

    print("\nOriginal Query:")
    print(user_query)

    rewritten = rewrite_query(user_query)
    print("\nRewritten Query:")
    print(rewritten)

    expanded = expand_query(rewritten, SYNONYM_MAP)
    print("\nExpanded Query:")
    print(expanded)

    multi_queries = generate_multi_queries(rewritten)
    print("\nMulti-Queries:")
    for idx, q in enumerate(multi_queries, start=1):
        print(f"{idx}. {q}")
