# eval.py
# Evaluation script for the SIC/SOC RAG classifier.
#
# Reads classification_results.csv and computes:
#   - Overall accuracy (on cases with known correct codes)
#   - Ambiguous rate  (% of valid inputs that went AMBIGUOUS)
#   - Human review rate
#   - Average attempts per classification
#   - Confidence distribution (HIGH / MEDIUM / AMBIGUOUS breakdown)
#
# In a production setting this would run on a labelled test set and
# feed results into a monitoring dashboard. Here it runs on whatever
# classify.py has produced so far.

import os
import csv
import json
from collections import Counter

RESULTS_PATH = "data/classification_results.csv"
AUDIT_PATH   = "data/audit_log.jsonl"

# ── Known correct codes for accuracy calculation ───────────────────────────────
# These are ground-truth labels for the non-vague test inputs.
# In production you'd have a labelled test dataset.
# SIC codes verified against ONS SIC 2007 condensed list.
KNOWN_CORRECT = {
    "software developer writing Python code for web applications":    "62010",
    "GP doctor working in an NHS health centre seeing patients":       "86210",
    "primary school teacher delivering lessons to children aged 7-11": "85200",
    "electrician installing wiring and consumer units in new build homes": "43210",
    "HGV lorry driver delivering goods between distribution centres":  "49410",
    "restaurant manager overseeing kitchen and front of house staff":  "56101",
    "nurse providing care on a hospital ward for elderly patients":     "86900",
    "accountant preparing tax returns for small businesses":           "69201",
    "civil servant processing Universal Credit benefit claims":        "84120",
}


def load_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        print("Run classify.py first to generate results.")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(results: list[dict]) -> dict:
    """
    Computes evaluation metrics across all classification results.
    Filters out invalid_input outcomes before computing rates —
    those are input quality issues, not model performance issues.
    """
    # Separate valid attempts from guardrail rejections
    valid_results = [
        r for r in results
        if r["outcome"] not in ("invalid_input", "retrieval_failed")
    ]
    invalid_count = len(results) - len(valid_results)

    if not valid_results:
        return {"error": "No valid classification results found"}

    # Outcome counts
    outcome_counts = Counter(r["outcome"] for r in valid_results)
    confidence_counts = Counter(
        r["confidence"] for r in valid_results if r["confidence"]
    )

    # Attempt stats
    attempt_values = [
        int(r["attempts"]) for r in valid_results if r["attempts"].isdigit()
    ]
    avg_attempts = sum(attempt_values) / len(attempt_values) if attempt_values else 0
    retry_count  = sum(1 for a in attempt_values if a > 1)

    # Accuracy on known cases
    accuracy_results = []
    for r in valid_results:
        correct_code = KNOWN_CORRECT.get(r["input_text"])
        if correct_code:
            is_correct = r["predicted_sic"] == correct_code
            accuracy_results.append({
                "input":        r["input_text"][:60],
                "predicted":    r["predicted_sic"],
                "correct":      correct_code,
                "is_correct":   is_correct,
                "confidence":   r["confidence"],
                "outcome":      r["outcome"]
            })

    correct_count = sum(1 for a in accuracy_results if a["is_correct"])
    accuracy = correct_count / len(accuracy_results) if accuracy_results else None

    # Rates
    total_valid = len(valid_results)
    classified_count   = outcome_counts.get("classified", 0)
    human_review_count = outcome_counts.get("human_review", 0)

    return {
        "total_inputs":         len(results),
        "invalid_inputs":       invalid_count,
        "valid_attempts":       total_valid,
        "classified":           classified_count,
        "human_review":         human_review_count,
        "classification_rate":  round(classified_count / total_valid * 100, 1),
        "human_review_rate":    round(human_review_count / total_valid * 100, 1),
        "avg_attempts":         round(avg_attempts, 2),
        "retry_triggered":      retry_count,
        "confidence_dist":      dict(confidence_counts),
        "accuracy_on_known":    round(accuracy * 100, 1) if accuracy is not None else "N/A",
        "accuracy_detail":      accuracy_results
    }


def print_report(metrics: dict):
    """Prints a formatted evaluation report to the console."""
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print("\n" + "="*65)
    print("SIC/SOC RAG CLASSIFIER — EVALUATION REPORT")
    print("="*65)

    print(f"\n── Input Summary ──────────────────────────────────────────")
    print(f"  Total inputs processed:   {metrics['total_inputs']}")
    print(f"  Invalid / rejected:       {metrics['invalid_inputs']}")
    print(f"  Valid classification runs: {metrics['valid_attempts']}")

    print(f"\n── Classification Outcomes ────────────────────────────────")
    print(f"  Classified:               {metrics['classified']}  "
          f"({metrics['classification_rate']}%)")
    print(f"  Escalated to human review:{metrics['human_review']}  "
          f"({metrics['human_review_rate']}%)")

    print(f"\n── Confidence Distribution ────────────────────────────────")
    for level in ["HIGH", "MEDIUM", "AMBIGUOUS"]:
        count = metrics["confidence_dist"].get(level, 0)
        pct   = round(count / metrics["valid_attempts"] * 100, 1)
        bar   = "█" * int(pct / 5)
        print(f"  {level:<10} {count:>3}  ({pct:>5}%)  {bar}")

    print(f"\n── Agentic Retry Stats ────────────────────────────────────")
    print(f"  Avg LLM attempts per run:  {metrics['avg_attempts']}")
    print(f"  Runs that triggered retry: {metrics['retry_triggered']}")

    print(f"\n── Accuracy on Known Test Cases ───────────────────────────")
    if metrics["accuracy_on_known"] == "N/A":
        print("  No labelled test cases found in results.")
    else:
        print(f"  Accuracy: {metrics['accuracy_on_known']}%")
        print()
        for r in metrics["accuracy_detail"]:
            status = "✓" if r["is_correct"] else "✗"
            print(f"  {status} [{r['confidence']:<9}] "
                  f"predicted={r['predicted']} correct={r['correct']}")
            print(f"    {r['input'][:60]}")

    print("\n" + "="*65)
    print("Langfuse dashboard: https://cloud.langfuse.com")
    print("="*65 + "\n")


def save_report_json(metrics: dict):
    """Saves the evaluation report as JSON for record-keeping."""
    report_path = "data/eval_report.json"
    # Remove detail list for clean top-level JSON
    report = {k: v for k, v in metrics.items() if k != "accuracy_detail"}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    print("Loading classification results...")
    results = load_results(RESULTS_PATH)

    if not results:
        exit(1)

    print(f"Loaded {len(results)} results. Computing metrics...")
    metrics = compute_metrics(results)
    print_report(metrics)
    save_report_json(metrics)