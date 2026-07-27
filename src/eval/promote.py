import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
ROUGE_RESULTS_PATH = REPO_ROOT / "eval_results_rouge.json"
GEMINI_SUMMARY_PATH = REPO_ROOT / "eval_summary_gemini_judge.json"
CHAMPION_SCORES_PATH = REPO_ROOT / "champion_scores.json"
PROMOTION_LOG_PATH = REPO_ROOT / "promotion_decisions.jsonl"

DEFAULT_THRESHOLD = 0.02
ROUGE_WEIGHT = 0.5
GEMINI_WEIGHT = 0.5


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required score file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rouge(scores: dict) -> float:
    metrics = ["rouge1", "rouge2", "rougeL"]
    values = [float(scores[name]) for name in metrics if name in scores]
    if not values:
        raise ValueError("ROUGE results must include at least one rouge metric.")
    return sum(values) / len(values)


def normalize_gemini(summary: dict) -> float:
    if "overall_average" not in summary:
        raise ValueError("Gemini summary must include overall_average.")
    return float(summary["overall_average"]) / 5.0


def build_scorecard(rouge_scores: dict, gemini_summary: dict) -> dict:
    rouge_average = normalize_rouge(rouge_scores)
    gemini_average = normalize_gemini(gemini_summary)
    combined = (ROUGE_WEIGHT * rouge_average) + (GEMINI_WEIGHT * gemini_average)
    return {
        "rouge_average": round(rouge_average, 6),
        "gemini_average": round(gemini_average, 6),
        "combined_score": round(combined, 6),
        "raw": {
            "rouge": rouge_scores,
            "gemini": gemini_summary,
        },
    }


def load_champion_scores(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return load_json(path)


def promotion_decision(
    champion: dict | None, challenger: dict, threshold: float
) -> dict:
    champion_score = None if champion is None else float(champion["combined_score"])
    challenger_score = float(challenger["combined_score"])

    promoted = champion_score is None or challenger_score > champion_score + threshold
    margin = (
        None if champion_score is None else round(challenger_score - champion_score, 6)
    )
    reason = (
        "No cached champion exists."
        if champion is None
        else "Compared against cached champion."
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PROMOTED" if promoted else "REJECTED",
        "reason": reason,
        "threshold": threshold,
        "margin": margin,
        "champion": champion,
        "challenger": challenger,
    }


def append_decision_log(path: Path, decision: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision) + "\n")


def save_champion(path: Path, challenger: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(challenger, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a challenger model when it beats the cached champion."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum normalized score improvement required for promotion.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write champion_scores.json and append promotion_decisions.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rouge_scores = load_json(ROUGE_RESULTS_PATH)
    gemini_summary = load_json(GEMINI_SUMMARY_PATH)
    challenger = build_scorecard(rouge_scores, gemini_summary)
    champion = load_champion_scores(CHAMPION_SCORES_PATH)
    decision = promotion_decision(champion, challenger, args.threshold)

    if args.update:
        append_decision_log(PROMOTION_LOG_PATH, decision)
        if decision["decision"] == "PROMOTED":
            save_champion(CHAMPION_SCORES_PATH, challenger)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
