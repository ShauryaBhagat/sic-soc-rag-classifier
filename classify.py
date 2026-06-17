# classify.py
# LLM-powered SIC code classifier with confidence scoring, agentic retry,
# and Langfuse v4 observability using the @observe decorator pattern.
#
# Langfuse v3+ is OpenTelemetry-native. The old langfuse.trace() API is gone.
# We now use @observe decorators — each decorated function automatically becomes
# a named span in the trace. Child spans nest correctly via OTel context propagation.
#
# Trace structure produced:
#   classify()                    ← root trace
#     ├── _validate_input()       ← span: input guardrail
#     ├── _retrieve_candidates()  ← span: vector retrieval
#     ├── _call_llm()             ← span: llm attempt 1 (generation)
#     ├── _call_llm()             ← span: llm attempt 2 retry (if triggered)
#     └── _check_output()        ← span: output guardrail

import os
import json
import re
import csv
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import observe, get_client
from retrieve import retrieve_top_n, validate_input

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME       = "gpt-4o-mini"
MAX_RETRIES      = 2
AUDIT_LOG_PATH   = "data/audit_log.jsonl"
RESULTS_LOG_PATH = "data/classification_results.csv"

# Langfuse client — used only for scoring (not for trace creation)
langfuse_client = get_client()

CONFIDENCE_SCORES = {"HIGH": 1.0, "MEDIUM": 0.5, "AMBIGUOUS": 0.0}


# ── Valid SIC codes master set ─────────────────────────────────────────────────
def _load_valid_sic_codes() -> set[str]:
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
        return set()

VALID_SIC_CODES = _load_valid_sic_codes()


# ── Prompt builders ────────────────────────────────────────────────────────────
def build_classification_prompt(input_text: str, candidates: list[dict]) -> str:
    candidates_str = "\n".join([
        f"  {i+1}. SIC {c['sic_code']}: {c['best_matching_activity']}"
        for i, c in enumerate(candidates)
    ])
    return f"""You are an expert in UK Standard Industrial Classification (SIC) coding.

Your task is to select the most appropriate SIC code for the following description.

INPUT DESCRIPTION:
"{input_text}"

CANDIDATE SIC CODES (retrieved by semantic search):
{candidates_str}

INSTRUCTIONS:
- Select the single best matching SIC code from the candidates above.
- If one candidate clearly fits, assign HIGH confidence.
- If a candidate fits but you have some uncertainty, assign MEDIUM confidence.
- If no candidate fits well, or two are equally plausible, assign AMBIGUOUS.
- Never invent a SIC code not in the candidate list.

Respond in this EXACT format — no extra text, no markdown:
SIC_CODE: [5-digit code]
CONFIDENCE: [HIGH|MEDIUM|AMBIGUOUS]
REASONING: [one sentence explaining your choice]"""


def build_retry_prompt(
    input_text: str, candidates: list[dict], attempt: int
) -> str:
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
1. What is the core economic activity described?
2. Which candidate SIC code best represents that activity?
3. What distinguishes this from the other candidates?
4. How confident are you?

After reasoning, respond in this EXACT format — no extra text:
SIC_CODE: [5-digit code]
CONFIDENCE: [HIGH|MEDIUM|AMBIGUOUS]
REASONING: [one sentence explaining your final choice]"""


# ── LLM response parser ────────────────────────────────────────────────────────
def parse_llm_response(response_text: str) -> dict | None:
    result = {}
    code_match   = re.search(r"SIC_CODE:\s*([0-9A-Za-z/]+)", response_text)
    conf_match   = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|AMBIGUOUS)", response_text)
    reason_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response_text, re.DOTALL)

    if code_match:   result["sic_code"]   = code_match.group(1).strip()
    if conf_match:   result["confidence"] = conf_match.group(1).strip()
    if reason_match: result["reasoning"]  = reason_match.group(1).strip()

    if not all(k in result for k in ["sic_code", "confidence", "reasoning"]):
        return None
    return result


# ── Instrumented sub-functions (each becomes a span in Langfuse) ───────────────

@observe(name="input_validation", as_type="span")
def _validate_input(input_text: str) -> tuple[bool, str]:
    """
    Span: input guardrail.
    @observe automatically logs the input and return value to Langfuse,
    and records how long this step took.
    """
    return validate_input(input_text)


@observe(name="vector_retrieval", as_type="span")
def _retrieve_candidates(input_text: str) -> list[dict] | None:
    """
    Span: vector similarity search.
    Logged as a child span of whichever classify() call invoked it.
    """
    return retrieve_top_n(input_text, top_n=5)


@observe(name="llm_call", as_type="generation")
def _call_llm(prompt: str, attempt_num: int) -> dict | None:
    """
    Generation span: one LLM call.
    as_type="generation" tells Langfuse this is an LLM call —
    it gets special treatment in the UI (token counts, model name, latency).
    Each attempt in the agentic loop creates its own generation span.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    response = llm.invoke(prompt)
    raw_text = response.content
    parsed = parse_llm_response(raw_text)
    # Return both raw and parsed so the span captures full LLM output
    return {
        "raw_response": raw_text,
        "parsed":       parsed,
        "attempt":      attempt_num
    }


@observe(name="output_guardrail", as_type="span")
def _check_output(predicted_code: str) -> tuple[bool, str]:
    """
    Span: output validation against ONS master list.
    Logged as a child span — visible in the trace tree in Langfuse.
    """
    if not VALID_SIC_CODES:
        return True, "validation_skipped"
    if predicted_code in VALID_SIC_CODES:
        return True, "valid"
    return False, f"Code '{predicted_code}' not in ONS SIC 2007 master list"


# ── Audit logging (not instrumented — these are file writes, not AI steps) ─────
def _write_audit_log(record: dict):
    os.makedirs("data", exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _write_results_csv(record: dict):
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
            "timestamp":     record["timestamp"],
            "input_text":    record["input_text"],
            "predicted_sic": record.get("final_code", ""),
            "confidence":    record.get("confidence", ""),
            "reasoning":     record.get("reasoning", ""),
            "attempts":      record["attempts"],
            "outcome":       record["outcome"]
        })


# ── Root trace — the @observe decorator here creates the top-level trace ───────
@observe(name="sic-classification")
def classify(input_text: str) -> dict:
    """
    Root trace function. The @observe decorator on this function creates
    the top-level Langfuse trace. All @observe-decorated functions called
    from within here automatically nest as child spans via OTel context
    propagation — no manual parent/child wiring needed.

    Returns a result dict with keys:
        outcome    : 'classified' | 'human_review' | 'invalid_input' | 'retrieval_failed'
        final_code : predicted SIC code (if classified)
        confidence : HIGH | MEDIUM | AMBIGUOUS
        reasoning  : LLM's explanation
        attempts   : number of LLM calls made
        candidates : retrieved SIC codes considered
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Input validation span ─────────────────────────────────
    is_valid, reason = _validate_input(input_text)

    if not is_valid:
        result = {
            "timestamp":  timestamp,
            "input_text": input_text,
            "outcome":    "invalid_input",
            "reason":     reason,
            "attempts":   0,
            "candidates": [],
            "final_code": "",
            "confidence": "",
            "reasoning":  ""
        }
        _write_audit_log(result)
        return result

    # ── Step 2: Retrieval span ─────────────────────────────────────────
    candidates = _retrieve_candidates(input_text)

    if not candidates:
        result = {
            "timestamp":  timestamp,
            "input_text": input_text,
            "outcome":    "retrieval_failed",
            "attempts":   0,
            "candidates": [],
            "final_code": "",
            "confidence": "",
            "reasoning":  ""
        }
        _write_audit_log(result)
        return result

    # ── Steps 3–5: Agentic LLM loop ───────────────────────────────────
    attempts    = 0
    last_parsed = None

    for attempt_num in range(1, MAX_RETRIES + 1):
        attempts += 1

        # Build prompt — chain-of-thought on retries
        if attempt_num == 1:
            prompt = build_classification_prompt(input_text, candidates)
        else:
            print(f"  [AGENT] Attempt {attempt_num}: retrying with chain-of-thought...")
            prompt = build_retry_prompt(input_text, candidates, attempt_num)

        # Each call creates a child generation span in Langfuse
        llm_result = _call_llm(prompt, attempt_num)
        parsed     = llm_result.get("parsed") if llm_result else None

        if parsed is None:
            print(f"  [AGENT] Could not parse LLM response on attempt {attempt_num}")
            continue

        last_parsed = parsed
        confidence  = parsed.get("confidence", "AMBIGUOUS")

        if confidence in ("HIGH", "MEDIUM"):
            break
        else:
            if attempt_num < MAX_RETRIES:
                print(f"  [AGENT] Confidence AMBIGUOUS on attempt {attempt_num}. Retrying...")
            else:
                print(
                    f"  [AGENT] Still AMBIGUOUS after {MAX_RETRIES} attempts. "
                    "Escalating to human review."
                )

    # ── Step 6: Determine outcome ──────────────────────────────────────
    if last_parsed is None:
        outcome    = "human_review"
        final_code = ""
        confidence = "AMBIGUOUS"
        reasoning  = "LLM response could not be parsed after all attempts"

    elif last_parsed.get("confidence") == "AMBIGUOUS":
        outcome    = "human_review"
        final_code = last_parsed.get("sic_code", "")
        confidence = "AMBIGUOUS"
        reasoning  = last_parsed.get("reasoning", "")

    else:
        final_code = last_parsed["sic_code"]
        confidence = last_parsed["confidence"]
        reasoning  = last_parsed["reasoning"]

        # ── Step 7: Output guardrail span ──────────────────────────────
        is_valid_out, validation_msg = _check_output(final_code)

        if not is_valid_out:
            print(f"  [GUARDRAIL] Output rejected: {validation_msg}")
            outcome = "human_review"
        else:
            outcome = "classified"

    # ── Step 8: Confidence score ───────────────────────────────────────
    # langfuse_client.score() attaches a numeric metric to the current trace.
    # This lets you plot confidence distribution in the Langfuse dashboard.
    try:
        langfuse_client.score(
            name="confidence_score",
            value=CONFIDENCE_SCORES.get(confidence, 0.0),
            comment=f"Confidence: {confidence} | Attempts: {attempts}"
        )
    except Exception:
        pass   # never let scoring failure crash a classification

    # ── Step 9: Audit log ──────────────────────────────────────────────
    result = {
        "timestamp":  timestamp,
        "input_text": input_text,
        "candidates": candidates,
        "attempts":   attempts,
        "final_code": final_code,
        "confidence": confidence,
        "reasoning":  reasoning,
        "outcome":    outcome
    }
    _write_audit_log(result)
    _write_results_csv(result)

    return result


# ── Pretty printer ─────────────────────────────────────────────────────────────
def print_result(result: dict):
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

    test_inputs = [
        "software developer writing Python code for web applications",
        "GP doctor working in an NHS health centre seeing patients",
        "primary school teacher delivering lessons to children aged 7-11",
        "electrician installing wiring and consumer units in new build homes",
        "HGV lorry driver delivering goods between distribution centres",
        "analyst working with large datasets and statistical models",
        "restaurant manager overseeing kitchen and front of house staff",
        "civil servant processing Universal Credit benefit claims",
        "nurse providing care on a hospital ward for elderly patients",
        "accountant preparing tax returns for small businesses",
        "manager overseeing a team of people",
        "works in an office doing admin",
        "self-employed consultant advising businesses",
        "works with computers",
        "coding free-text survey responses about people's jobs",
        "data scientist building NLP classification models for surveys",
        "",
        "xyz",
        "99999 00000",
    ]

    print(f"Running {len(test_inputs)} classifications with Langfuse v4 tracing...")
    print(f"View traces at: https://cloud.langfuse.com\n")

    classified   = 0
    human_review = 0
    invalid      = 0

    for text in test_inputs:
        result = classify(text)
        print_result(result)
        if result["outcome"] == "classified":       classified   += 1
        elif result["outcome"] == "human_review":   human_review += 1
        else:                                        invalid      += 1

    # flush() is still needed in v4 for short-lived scripts —
    # ensures all queued spans are exported before the process exits
    langfuse_client.flush()

    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"  Classified:    {classified}")
    print(f"  Human review:  {human_review}")
    print(f"  Invalid input: {invalid}")
    print(f"  Total:         {len(test_inputs)}")
    print(f"\nTraces visible at: https://cloud.langfuse.com")