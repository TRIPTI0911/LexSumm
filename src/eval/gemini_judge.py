import os
import json
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
import time
MAX_EXAMPLES = 30  # reduced to avoid quota exhaustion
# Load .env to get the Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env. Please add it before running the judge script.")

# Configure the Gemini client
client = genai.Client(api_key=api_key)
# Allow overriding model via env var (default gemini-3.5-flash)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# System prompt for the judge – can be edited if a different evaluation style is desired
JUDGE_PROMPT = (
    "You are an expert legal‑summarization evaluator. Given a reference summary and a model‑generated "
    "prediction, rate the prediction on three criteria: relevance, factual correctness, and fluency, each "
    "from 1 (worst) to 5 (best). Return ONLY a JSON object with EXACTLY these keys, no others: "
    "{\"relevance\": <int>, \"factual\": <int>, \"fluency\": <int>, "
    "\"relevance_reasoning\": \"<one sentence>\", \"factual_reasoning\": \"<one sentence>\", \"fluency_reasoning\": \"<one sentence>\"}"
    )





def evaluate() -> list:
    """Fetch the prediction dataset from HF Hub and evaluate each example with Gemini.
    The dataset uses the 'train' split (the 200 test predictions).
    This script limits evaluation to a safe number of examples (MAX_EXAMPLES) and
    performs exponential backoff retries on rate‑limit errors.
    The judge prompt now also asks for a brief reasoning for each score.
    """
    ds = load_dataset("Tripti0911/lexsumm-predictions-v1")["train"]
    # Limit to MAX_EXAMPLES for quota safety
    if len(ds) > MAX_EXAMPLES:
        ds = ds.select(range(MAX_EXAMPLES))
    results = []
    for idx, example in enumerate(ds):
        # Build the user message that includes both reference and prediction
        user_message = (
            f"Reference summary:\n{example['reference_summary']}\n\n"
            f"Model prediction:\n{example['generated_summary']}"
        )
        # Send the prompt + user message to Gemini
        # Simple retry with exponential backoff for rate limiting
        attempts = 0
        while True:
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=JUDGE_PROMPT,
                    ),
                )
                break
            except Exception as e:
                # Only retry on rate‑limit errors (HTTP 429). Propagate other issues.
                if "429" not in str(e) and "rate" not in str(e).lower():
                    raise
                attempts += 1
                if attempts > 5:
                    # Quota likely exhausted – break out and stop further calls
                    print("[WARN] Quota exhausted or max retries reached; stopping evaluation early.")
                    # Return whatever results we have so far
                    return results
                # Wait longer each retry (exponential backoff)
                time.sleep(2 ** attempts)
        # Try to parse the JSON answer; handle possible markdown fences
        raw_text = response.text.strip()
        # Strip generic markdown fences (``` or ```json) if present
        if raw_text.startswith("```"):
            # Remove opening fence (may include language spec)
            parts = raw_text.split("\n", 1)
            raw_text = parts[1] if len(parts) > 1 else ""
            # Remove closing fence if it exists
            if raw_text.endswith("```"):
                raw_text = raw_text.rsplit("\n", 1)[0]
        try:
            scores = json.loads(raw_text)
        except Exception:
            scores = {"error": raw_text}
        # Merge scores with the original example identifiers
        merged = {
            "id": example.get("id", idx),
            "reference_summary": example["reference_summary"],
            "generated_summary": example["generated_summary"],
            **scores,
        }
        results.append(merged)
    return results


if __name__ == "__main__":
    evaluation_results = evaluate()
    # Save evaluation results to a JSON file for later use
    with open("eval_results_gemini_judge.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    # Print as pretty JSON for easy downstream consumption
    print(json.dumps(evaluation_results, indent=2))
