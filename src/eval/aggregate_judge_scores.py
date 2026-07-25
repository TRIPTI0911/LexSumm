import json
import os
from pathlib import Path

# Paths – assume script runs from repo root
RESULTS_PATH = Path(__file__).parents[2] / "eval_results_gemini_judge.json"
SUMMARY_PATH = Path(__file__).parents[2] / "eval_summary_gemini_judge.json"

def load_results():
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Gemini judge results not found at {RESULTS_PATH}")
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_summary(results):
    # Expect each result dict to contain integer keys: relevance, factual, fluency
    if not results:
        raise ValueError("No evaluation results to aggregate.")
    total = {"relevance": 0, "factual": 0, "fluency": 0}
    count = 0
    for r in results:
        try:
            total["relevance"] += int(r.get("relevance", 0))
            total["factual"] += int(r.get("factual", 0))
            total["fluency"] += int(r.get("fluency", 0))
            count += 1
        except (ValueError, TypeError):
            continue
    if count == 0:
        raise ValueError("No valid scores found in results.")
    avg = {k: round(v / count, 3) for k, v in total.items()}
    overall = round(sum(avg.values()) / 3, 3)
    return {
        "mean_relevance": avg["relevance"],
        "mean_factual": avg["factual"],
        "mean_fluency": avg["fluency"],
        "overall_average": overall,
        "num_examples": count,
    }

def main():
    results = load_results()
    summary = compute_summary(results)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
