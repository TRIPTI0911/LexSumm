import json

from datasets import Dataset, load_dataset
from rouge_score import rouge_scorer, scoring


def load_predictions() -> Dataset:
    """Load the prediction dataset from the HF Hub.

    The dataset was pushed using the ``train`` split (contains the 200 test predictions).
    Each example has the fields ``reference_summary`` and ``generated_summary``.
    """
    ds = load_dataset("Tripti0911/lexsumm-predictions-v1")
    # The predictions were pushed as the "train" split (contains 200 test examples)
    return ds["train"]


def compute_rouge(dataset: Dataset) -> dict:
    """Compute ROUGE‑1, ROUGE‑2 and ROUGE‑L F1 scores.

    The function uses ``rouge_score``'s ``RougeScorer`` for per‑example scores
    and ``BootstrapAggregator`` to aggregate them into a single mid‑point value.
    It works with the ``reference_summary`` and ``generated_summary`` fields produced
    by the prediction script.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for example in dataset:
        ref = example["reference_summary"]
        pred = example["generated_summary"]
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)

    aggregated = aggregator.aggregate()
    # Return only the mid F‑measure for each metric
    return {metric: values.mid.fmeasure for metric, values in aggregated.items()}


if __name__ == "__main__":
    ds = load_predictions()
    rouge_scores = compute_rouge(ds)
    with open("eval_results_rouge.json", "w") as f:
        json.dump(rouge_scores, f, indent=2)
    print(json.dumps(rouge_scores, indent=2))
