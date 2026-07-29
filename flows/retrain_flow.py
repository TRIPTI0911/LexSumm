import argparse
from pathlib import Path

from prefect import flow, get_run_logger, task

REPO_ROOT = Path(__file__).parents[1]
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.jsonl"
VAL_PATH = REPO_ROOT / "data" / "processed" / "val.jsonl"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.jsonl"


@task
def ensure_processed_dataset(force_prepare: bool = False) -> None:
    logger = get_run_logger()
    required_splits = [TRAIN_PATH, VAL_PATH, TEST_PATH]
    missing = [path for path in required_splits if not path.is_file()]
    if not force_prepare and not missing:
        logger.info("Processed dataset already exists.")
        return

    if force_prepare:
        logger.info("Preparing dataset because --force-prepare was provided.")
    else:
        logger.info(
            "Preparing dataset because processed splits are missing: %s", missing
        )

    from src.data_prep import main as data_prep_main

    data_prep_main()


@task
def kaggle_training_handoff() -> None:
    logger = get_run_logger()
    logger.info(
        "Manual handoff: upload the processed dataset to Kaggle, run src/train.py "
        "on a GPU notebook, push the adapter/GGUF artifact to Hugging Face Hub, "
        "then rerun this flow with --resume to evaluate and promote."
    )


@task
def evaluate_model() -> None:
    logger = get_run_logger()
    logger.info("Aggregating Gemini judge results.")
    from src.eval.aggregate_judge_scores import main as aggregate_judge_scores_main

    aggregate_judge_scores_main()


@task
def publish_model() -> None:
    logger = get_run_logger()
    logger.info("Running local champion/challenger promotion gate.")
    from src.eval.promote import (
        DEFAULT_THRESHOLD,
        GEMINI_SUMMARY_PATH,
        ROUGE_RESULTS_PATH,
        build_scorecard,
        load_champion_scores,
        load_json,
        promotion_decision,
    )

    rouge_scores = load_json(ROUGE_RESULTS_PATH)
    gemini_summary = load_json(GEMINI_SUMMARY_PATH)
    challenger = build_scorecard(rouge_scores, gemini_summary)
    champion = load_champion_scores(REPO_ROOT / "champion_scores.json")
    decision = promotion_decision(champion, challenger, DEFAULT_THRESHOLD)
    logger.info("Promotion decision: %s", decision["decision"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LexSumm Prefect pipeline.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Run post-Kaggle evaluation and promotion after training is complete.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Regenerate processed dataset splits before the Kaggle training handoff.",
    )
    return parser.parse_args()


@flow(name="lexsumm-retraining-pipeline")
def retraining_pipeline(resume: bool = False, force_prepare: bool = False) -> None:
    logger = get_run_logger()
    if not resume:
        ensure_processed_dataset(force_prepare)
        kaggle_training_handoff()
        logger.info(
            "Pipeline paused before evaluation. Rerun with --resume after Kaggle training."
        )
        return

    evaluate_model()
    publish_model()


if __name__ == "__main__":
    args = parse_args()
    retraining_pipeline(resume=args.resume, force_prepare=args.force_prepare)
