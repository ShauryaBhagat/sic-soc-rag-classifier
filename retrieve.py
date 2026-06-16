# retrieve.py
# Retrieves the top-n most relevant SIC codes for a given text description.
# Searches across 15,958 activity descriptions and returns the best matching
# SIC codes with similarity scores.
# Includes input guardrails and audit logging — demonstrates governance-aware design.

import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "data/chroma_store"
DEFAULT_TOP_N = 5
MIN_INPUT_LENGTH = 5
MAX_INPUT_LENGTH = 500
REJECTED_LOG_PATH = "data/rejected_inputs.log"


def validate_input(text: str) -> tuple[bool, str]:
    """
    Guardrail: validates input before it reaches the embedding model.
    Rejects empty, too-short, too-long, and numeric-only inputs.

    Why this matters: input validation is the first line of defence in
    production AI systems — it prevents wasted API calls, prompt injection
    attempts, and nonsense results propagating downstream.
    """
    if not text or not text.strip():
        return False, "Input is empty or whitespace only"

    text = text.strip()

    if len(text) < MIN_INPUT_LENGTH:
        return False, f"Input too short (minimum {MIN_INPUT_LENGTH} characters)"

    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Input too long (maximum {MAX_INPUT_LENGTH} characters)"

    if re.fullmatch(r"[\d\s\-\.\,]+", text):
        return False, "Input appears to be numeric only — expected a text description"

    return True, "valid"


def log_rejected_input(text: str, reason: str):
    """
    Logs rejected inputs with timestamp for audit and monitoring.
    In production this would feed into an observability dashboard.
    """
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    with open(REJECTED_LOG_PATH, "a") as f:
        f.write(f"{timestamp} | REJECTED | reason='{reason}' | input='{text[:100]}'\n")


def load_vector_store() -> Chroma:
    """Loads the persisted ChromaDB store from disk."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_PATH}. "
            "Please run ingest.py first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


def deduplicate_by_code(results: list[tuple]) -> list[dict]:
    """
    The ONS file has many activity descriptions per SIC code.
    A search may return multiple activities from the same code.
    This deduplicates to return the best-matching activity per unique SIC code,
    so the user sees clean distinct candidates rather than repetitions.
    """
    seen_codes = {}
    for doc, score in results:
        code = doc.metadata["sic_code"]
        if code not in seen_codes:
            seen_codes[code] = {
                "sic_code": code,
                "best_matching_activity": doc.metadata["activity"],
                "similarity_score": round(float(score), 4)
            }
        # If we've seen this code before, keep whichever has lower distance score
        elif float(score) < seen_codes[code]["similarity_score"]:
            seen_codes[code]["best_matching_activity"] = doc.metadata["activity"]
            seen_codes[code]["similarity_score"] = round(float(score), 4)

    return list(seen_codes.values())


def retrieve_top_n(text: str, top_n: int = DEFAULT_TOP_N) -> list[dict] | None:
    """
    Main retrieval function.
    1. Validates input (guardrail)
    2. Embeds the query text via OpenAI
    3. Searches ChromaDB for most similar activity descriptions
    4. Deduplicates by SIC code
    5. Returns top_n unique SIC code candidates with scores

    Returns None if input fails validation.
    """
    # Step 1: Input guardrail
    is_valid, reason = validate_input(text)
    if not is_valid:
        log_rejected_input(text, reason)
        print(f"[GUARDRAIL] Input rejected: {reason}")
        return None

    # Steps 2 & 3: Embed and search
    # We fetch top_n * 5 raw results to ensure we have enough
    # unique SIC codes after deduplication
    store = load_vector_store()
    raw_results = store.similarity_search_with_score(
        text.strip(),
        k=top_n * 5
    )

    # Step 4: Deduplicate by SIC code
    candidates = deduplicate_by_code(raw_results)

    # Step 5: Return top_n after deduplication
    return candidates[:top_n]


def print_results(query: str, candidates: list[dict]):
    """Pretty-prints retrieval results for testing."""
    print(f"\nQuery: '{query}'")
    print("-" * 65)
    if not candidates:
        print("No results (input failed validation).")
        return
    for i, c in enumerate(candidates, 1):
        print(f"{i}. SIC {c['sic_code']}")
        print(f"   Best match: {c['best_matching_activity']}")
        print(f"   Distance:   {c['similarity_score']}  (lower = more similar)")
    print("-" * 65)


if __name__ == "__main__":
    test_inputs = [
        # Valid — range of occupations
        "software developer writing code for web applications",
        "nurse working in an NHS hospital providing patient care",
        "restaurant owner managing a takeaway food business",
        "electrician installing wiring in residential buildings",
        "data scientist analysing census survey responses",
        # Guardrail cases — should be rejected cleanly
        "",
        "abc",
        "12345 67890",
    ]

    for query in test_inputs:
        results = retrieve_top_n(query, top_n=5)
        if results:
            print_results(query, results)