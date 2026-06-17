# classify.py
# LLM-powered SIC code classifier with confidence scoring and agentic retry.
#
# Architecture:
#   1. Retrieve top-5 candidate SIC codes via vector search (retrieve.py)
#   2. Ask LLM to select the best match with a confidence level
#   3. Parse structured output (code, confidence, reasoning)
#   4. Agentic gate: if ambiguous, retry with chain-of-thought reasoning
#   5. After 2 failed attempts, escalate to human review
#   6. Log everything to a structured audit trail (JSON Lines format)
#
# This mirrors the ONS production LLM classification pipeline architecture.

import os
import json
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from retrieve import retrieve_top_n, validate_input

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "gpt-4o-mini"          # cheap, fast, capable enough for this task
MAX_RETRIES = 2                      # agentic loop: max attempts before escalation
AUDIT_LOG_PATH = "data/audit_log.jsonl"
RESULTS_LOG_PATH = "data/classification_results.csv"

# Valid SIC codes master set — loaded once at module level for output validation
def _load_valid_sic_codes() -> set[str]:
    """Loads all valid SIC codes from the ONS Excel for output guardrail."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook("data/SIC_2007_index.xlsx", read_only=True)
        ws = wb.active
        codes = set()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0]:
                codes.add(str(row[0]).strip())
        return codes
    except Exception:
        return set()   # graceful fallback — don't crash if file unavailable

VALID_SIC_CODES = _load_valid_sic_codes()


# ── Prompt templates ───────────────────────────────────────────────────────────

def build_classification_prompt(input_text: str, candidates: list[dict]) -> str:
    """
    Builds the first-attempt classification prompt.
    Injects retrieved candidates as context — this is the RAG pattern:
    the LLM's knowledge is grounded in retrieved evidence, not just
    its training data.
    """
    candidates_str = "\n".join([
        f"  {i+1}. SIC {c['sic_code']}: {c['best_matching_activity']}"
        for i, c in enumerate(candidates)
    ])

    return f"""You are an expert in UK Standard Industrial Classification (SIC) coding.

Your task is to select the most appropriate SIC code for the following description of a job or business activity.

INPUT DESCRIPTION:
"{input_text}"

CANDIDATE SIC CODES (retrieved by semantic search):
{candidates_str}

INSTRUCTIONS:
- Select the single best matching SIC code from the candidates above.
- If one candidate clearly fits, assign HIGH confidence.
- If a candidate fits but you have some uncertainty, assign MEDIUM confidence.
- If no candidate fits well, or two candidates are equally plausible, assign AMBIGUOUS.
- Never invent a SIC code not in the candidate list.

Respond in this EXACT format — no extra text, no markdown:
SIC_CODE: [5-digit code]
CONFIDENCE: [HIGH|MEDIUM|AMBIGUOUS]
REASONING: [one sentence explaining your choice]"""


def build_retry_prompt(input_text: str, candidates: list[dict], attempt: int) -> str:
    """
    Builds the retry prompt for the agentic loop.
    Uses chain-of-thought reasoning — asks the LLM to think step by step
    before committing. This technique reliably improves accuracy on
    ambiguous classification tasks.
    """
    candidates_str = "\n".join([
        f"  {i+1}. SIC {c['sic_code']}: {c['best_matching_activity']}"
        for i, c in enumerate(candidates)
    ])

    return f"""You are an expert in UK Standard Industrial Classification (SIC) coding.

A previous attempt to classify the following description was inconclusive.
This is retry attempt {attempt} of {MAX_RETRIES}. Please reason carefully.

INPUT DESCRIPTION:
"{input_text}"

CANDIDATE SIC CODES:
{candidates_str}

THINK STEP BY STEP:
1. What is the core economic activity described in the input?
2. Which candidate SIC code best represents that activity?
3. What distinguishes this from the other candidates?
4. How confident are you?

After reasoning, respond in this EXACT format — no extra text:
SIC_CODE: [5-digit code]
CONFIDENCE: [HIGH|MEDIUM|AMBIGUOUS]
REASONING: [one sentence explaining your final choice]"""


# ── LLM response parser ────────────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> dict | None:
    """
    Parses the structured LLM output into a dictionary.
    Uses regex to extract fields robustly — handles minor formatting
    variations in LLM output without crashing.

    Returns None if parsing fails entirely (triggers fallback logic).
    """
    result = {}

    # Extract SIC_CODE
    code_match = re.search(r"SIC_CODE:\s*([0-9A-Za-z/]+)", response_text)
    if code_match:
        result["sic_code"] = code_match.group(1).strip()

    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|AMBIGUOUS)", response_text)
    if conf_match:
        result["confidence"] = conf_match.group(1).strip()

    # Extract REASONING
    reason_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response_text, re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    # Validate we got all three fields
    if not all(k in result for k in ["sic_code", "confidence", "reasoning"]):
        return None

    return result


# ── Output guardrail ───────────────────────────────────────────────────────────

def validate_output(predicted_code: str) -> tuple[bool, str]:
    """
    Output guardrail: validates the LLM's predicted SIC code against
    the master ONS list.

    Why this matters: LLMs can hallucinate plausible-looking but invalid
    codes. In a production system an invalid code causes downstream
    failures in statistics pipelines. This check is the last line of
    defence before a result is accepted.
    """
    if not VALID_SIC_CODES:
        # Master list unavailable — skip validation with a warning
        return True, "validation_skipped"

    if predicted_code in VALID_SIC_CODES:
        return True, "valid"

    return False, f"Predicted code '{predicted_code}' not in ONS SIC 2007 master list"


# ── Audit logging ──────────────────────────────────────────────────────────────

def write_audit_log(record: dict):
    """
    Appends a structured audit record to a JSON Lines file.
    Each line is a valid JSON object — easy to parse, query, and
    feed into monitoring tools.

    Fields logged:
      - timestamp, input_text, candidates, attempts, final_code,
        confidence, reasoning, outcome, output_valid
    This provides full traceability for every classification decision —
    a key requirement for responsible AI in production.
    """
    os.makedirs("data", exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_results_csv(record: dict):
    """Appends the classification result to a CSV for easy review."""
    import csv
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(RESULTS_LOG_PATH)

    with open(RESULTS_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "input_text", "predicted_sic",
            "confidence", "reasoning", "attempts", "outcome"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": record["timestamp"],
            "input_text": record["input_text"],
            "predicted_sic": record.get("final_code", ""),
            "confidence": record.get("confidence", ""),
            "reasoning": record.get("reasoning", ""),
            "attempts": record["attempts"],
            "outcome": record["outcome"]
        })


# ── Core agentic classification loop ──────────────────────────────────────────

def classify(input_text: str) -> dict:
    """
    Main classification function — the agentic loop.

    This is the heart of the pipeline. It:
    1. Validates the input
    2. Retrieves candidate SIC codes
    3. Asks the LLM to classify
    4. Evaluates the LLM's confidence
    5. If ambiguous: retries with chain-of-thought (agentic behaviour)
    6. After MAX_RETRIES: escalates to human review
    7. Validates the output against the master SIC list
    8. Logs everything to the audit trail

    Returns a result dict with keys:
      outcome    : 'classified' | 'human_review' | 'retrieval_failed' | 'invalid_input'
      final_code : predicted SIC code (if classified)
      confidence : HIGH | MEDIUM | AMBIGUOUS
      reasoning  : LLM's explanation
      attempts   : number of LLM calls made
      candidates : the retrieved SIC codes considered
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    # temperature=0 → deterministic output, essential for reproducible
    # classification. Non-zero temperature introduces randomness that
    # would make audit logs unreproducible.

    # ── Step 1: Input validation ───────────────────────────────────────
    is_valid, reason = validate_input(input_text)
    if not is_valid:
        result = {
            "timestamp": timestamp,
            "input_text": input_text,
            "outcome": "invalid_input",
            "reason": reason,
            "attempts": 0,
            "candidates": []
        }
        write_audit_log(result)
        return result

    # ── Step 2: Retrieve candidates ────────────────────────────────────
    candidates = retrieve_top_n(input_text, top_n=5)
    if not candidates:
        result = {
            "timestamp": timestamp,
            "input_text": input_text,
            "outcome": "retrieval_failed",
            "attempts": 0,
            "candidates": []
        }
        write_audit_log(result)
        return result

    # ── Steps 3–5: Agentic classification loop ─────────────────────────
    attempts = 0
    last_parsed = None

    for attempt_num in range(1, MAX_RETRIES + 1):
        attempts += 1

        # Choose prompt — chain-of-thought on retries
        if attempt_num == 1:
            prompt = build_classification_prompt(input_text, candidates)
        else:
            print(f"  [AGENT] Attempt {attempt_num}: retrying with chain-of-thought...")
            prompt = build_retry_prompt(input_text, candidates, attempt_num)

        # Call the LLM
        response = llm.invoke(prompt)
        raw_text = response.content
        parsed = parse_llm_response(raw_text)

        if parsed is None:
            # Parsing failed entirely — treat as ambiguous and retry
            print(f"  [AGENT] Could not parse LLM response on attempt {attempt_num}")
            continue

        last_parsed = parsed
        confidence = parsed.get("confidence", "AMBIGUOUS")

        if confidence in ("HIGH", "MEDIUM"):
            # Confident prediction — exit the loop
            break
        else:
            # AMBIGUOUS — if we have retries left, loop again
            if attempt_num < MAX_RETRIES:
                print(f"  [AGENT] Confidence AMBIGUOUS on attempt {attempt_num}. Retrying...")
            else:
                print(f"  [AGENT] Still AMBIGUOUS after {MAX_RETRIES} attempts. "
                      "Escalating to human review.")

    # ── Step 6: Determine outcome ──────────────────────────────────────
    if last_parsed is None:
        outcome = "human_review"
        final_code = ""
        confidence = "AMBIGUOUS"
        reasoning = "LLM response could not be parsed after all attempts"
    elif last_parsed.get("confidence") == "AMBIGUOUS":
        outcome = "human_review"
        final_code = last_parsed.get("sic_code", "")
        confidence = "AMBIGUOUS"
        reasoning = last_parsed.get("reasoning", "")
    else:
        final_code = last_parsed["sic_code"]
        confidence = last_parsed["confidence"]
        reasoning = last_parsed["reasoning"]

        # ── Step 7: Output guardrail ───────────────────────────────────
        is_valid_output, validation_msg = validate_output(final_code)
        if not is_valid_output:
            print(f"  [GUARDRAIL] Output rejected: {validation_msg}")
            outcome = "human_review"
        else:
            outcome = "classified"

    # ── Step 8: Audit log ──────────────────────────────────────────────
    audit_record = {
        "timestamp": timestamp,
        "input_text": input_text,
        "candidates": candidates,
        "attempts": attempts,
        "final_code": final_code,
        "confidence": confidence,
        "reasoning": reasoning,
        "outcome": outcome
    }
    write_audit_log(audit_record)
    write_results_csv(audit_record)

    return audit_record


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_result(result: dict):
    """Prints a single classification result clearly to the console."""
    print(f"\n{'='*65}")
    print(f"INPUT:      {result['input_text']}")
    print(f"OUTCOME:    {result['outcome']}")
    if result.get("final_code"):
        print(f"SIC CODE:   {result['final_code']}")
    if result.get("confidence"):
        print(f"CONFIDENCE: {result['confidence']}")
    if result.get("reasoning"):
        print(f"REASONING:  {result['reasoning']}")
    print(f"ATTEMPTS:   {result.get('attempts', 0)}")
    if result.get("candidates"):
        print(f"\nCANDIDATES CONSIDERED:")
        for c in result["candidates"]:
            print(f"  SIC {c['sic_code']}: {c['best_matching_activity']}")
    print(f"{'='*65}")


# ── Test harness ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    # 20 test inputs covering a range of occupations and edge cases
    # Mix of clear cases (should get HIGH confidence) and ambiguous ones
    test_inputs = [
        # Clear cases — expect HIGH confidence
        "software developer writing Python code for web applications",
        "GP doctor working in an NHS health centre seeing patients",
        "primary school teacher delivering lessons to children aged 7-11",
        "electrician installing wiring and consumer units in new build homes",
        "HGV lorry driver delivering goods between distribution centres",

        # Moderately clear — expect MEDIUM or HIGH
        "analyst working with large datasets and statistical models",
        "restaurant manager overseeing kitchen and front of house staff",
        "civil servant processing Universal Credit benefit claims",
        "nurse providing care on a hospital ward for elderly patients",
        "accountant preparing tax returns for small businesses",

        # Ambiguous cases — expect AMBIGUOUS → agentic retry
        "manager overseeing a team of people",           # too vague
        "works in an office doing admin",                # too vague
        "self-employed consultant advising businesses",  # could be many codes
        "works with computers",                         # very vague

        # Your ONS domain — test whether the retrieval knows this world
        "coding free-text survey responses about people's jobs",
        "data scientist building NLP classification models for surveys",

        # Guardrail cases
        "",
        "xyz",
        "99999 00000",
    ]

    print(f"Running classification on {len(test_inputs)} test inputs...")
    print(f"Model: {MODEL_NAME} | Max retries: {MAX_RETRIES}")
    print(f"Audit log: {AUDIT_LOG_PATH}")
    print(f"Results CSV: {RESULTS_LOG_PATH}\n")

    classified = 0
    human_review = 0
    invalid = 0

    for text in test_inputs:
        result = classify(text)
        print_result(result)

        if result["outcome"] == "classified":
            classified += 1
        elif result["outcome"] == "human_review":
            human_review += 1
        else:
            invalid += 1

    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"  Classified:    {classified}")
    print(f"  Human review:  {human_review}")
    print(f"  Invalid input: {invalid}")
    print(f"  Total:         {len(test_inputs)}")
    print(f"\nFull audit log saved to: {AUDIT_LOG_PATH}")
    print(f"Results CSV saved to:    {RESULTS_LOG_PATH}")