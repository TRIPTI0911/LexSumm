# LLMOps Pipeline: Domain-Specific Summarization Model (Legal Domain)

This repository contains a production-grade, end-to-end LLMOps pipeline designed for fine-tuning, evaluating, serving, and monitoring a domain-specific summarization model (using BillSum for legal bill summarization). The system is built leveraging a **zero-cost free-tier stack** to showcase high maturity constraint-driven design decisions.

---

## Architecture Overview

The pipeline implements an automated training, validation, serving, and monitoring loop. Below is the system flow:

```mermaid
flowchart TD
    subgraph Data [1. Data Layer]
        A[Hugging Face Datasets: BillSum] -->|Download & Clean| B[src/data_prep.py]
        B -->|Subsample & Split| C[data/processed/]
        C -->|Upload| D[HF Dataset Hub / Data Registry]
    end

    subgraph FineTuning [2. Training Layer]
        C -->|Read splits| E[Kaggle T4 GPU x2]
        E -->|QLoRA: Unsloth| F[W&B: Run Tracking]
        E -->|Push Adapter| G[HF Model Hub / Model Registry]
    end

    subgraph Evaluation [3. Evaluation Gate]
        G -->|Fetch adapter| H[src/eval/rouge_eval.py]
        G -->|Fetch adapter| I[src/eval/gemini_judge.py]
        H -->|ROUGE / BERTScore| J[src/eval/promote.py]
        I -->|LLM-as-a-Judge| J
        J -->|Challenger beats Champ?| K{Promotion Gate}
        K -->|Yes| L[Tag HF model as main/champion]
        K -->|No| M[Reject Challenger]
        L -->|Log event| N[(Supabase Database)]
        M -->|Log event| N
    end

    subgraph Serving [4. Serving Layer]
        L -->|Merge & Quantize to GGUF| O[HF Hub GGUF Registry]
        O -->|Serve serverless| P[FastAPI serving/app.py]
        P -->|Deploy| Q[HF Spaces / Modal]
    end

    subgraph Monitoring [5. Observability]
        Q -->|Log metrics / predictions| N
        N -->|Query logs| R[Streamlit Dashboard]
        Q -->|Sample 10% traffic| I
    end

    style Data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    style FineTuning fill:#efebe9,stroke:#5d4037,stroke-width:2px;
    style Evaluation fill:#efe2ba,stroke:#f57c00,stroke-width:2px;
    style Serving fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    style Monitoring fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
```

---

## Technology Stack (Free-Tier Optimization)

| Layer | Tool | Rationale | Cost |
|---|---|---|---|
| **Fine-Tuning** | Unsloth + QLoRA | Fits 3B/4B models comfortably on free-tier T4 GPU (up to 2x faster, 70% memory reduction). | **$0** |
| **Experiment Tracking** | Weights & Biases | Free hosting for runs, config history, loss curves, and artifact tracking. | **$0** |
| **Model & Data Registry** | Hugging Face Hub | Host and version datasets & LoRA/GGUF model adapters natively. | **$0** |
| **Evaluation** | Local CPU + Gemini Flash | ROUGE/BERTScore compute locally. Gemini 2.0/2.5 Flash API serves as a low-cost, high-quality judge. | **$0** |
| **Serving** | HF Inference Endpoints / Spaces | Serverless scale-to-zero serving or free CPU container hosting for the API. | **$0** |
| **CI/CD** | GitHub Actions | Automatically run unit tests, check linting, and build/push FastAPI Docker image. | **$0** |
| **Orchestration** | Prefect (Local Process) | Lightweight automation. Triggers data checks and orchestrates Kaggle training via Kaggle API. | **$0** |
| **Monitoring** | Supabase + Streamlit | PostgreSQL logs requests (latency, inputs, outputs, cost). Streamlit hosts a public dashboard. | **$0** |

---

## Phase 1: Dataset Preparation

We use the US Congressional Bills **BillSum** dataset to fine-tune our summarizer. 

### Subsampling & Preprocessing Decisions (Why & How)

To build a reliable and budget-conscious fine-tuning dataset, we apply the following data preparation rules inside [src/data_prep.py](file:///Users/triptigupta/Desktop/LexSumm/src/data_prep.py):

1. **Length Constraint Filtering (The sequence budget):**
   * Pre-trained model context windows (such as Phi-3-mini or Llama-3.2) typically run with a 2048-token context window during training to optimize memory. 
   * A token is roughly 4 characters. To fit the instruction prompt, source text, and summary target within **2048 tokens**, we restrict text lengths to:
     * **Source text (Bill)**: between `1,500` and `6,000` characters.
     * **Summary**: between `200` and `1,500` characters.
   * If a bill is too short, it doesn't present a realistic summarization task. If it's too long, truncation will sever the tail-end of the bill and lead to incomplete summarizations.
2. **Quality Gate Cleaning:**
   * Removes congressional header boilerplate (e.g., sessions, H.R. numbers) using regular expressions.
   * Standardizes TeX-style quotes (normalizes `` ` `` `` and `''` to `"`) to keep tokenization clean.
   * Normalizes redundant whitespaces, newlines, and tabs.
   * **Near-Deduplication:** Prevents leakage of identical bills re-introduced in different congressional sessions with minor header edits. This is done by extracting the first ~800 characters of the bill, removing all non-alphanumeric characters, and dropping duplicates using the resulting normalized 500-character prefix hash.
   * Drops invalid/flipped examples. *(Note: Extractive shortcut checking is documented but omitted since BillSum summaries are professionally written abstracts.)*
3. **Deliberate Systematic Subsampling:**
   * To prevent sampling bias, we sort the cleaned and length-filtered corpus by character length and systematically extract **3,500** evenly-spaced examples.
   * This non-random approach guarantees that the training, validation, and test datasets represent an identical length distribution range of short, medium, and long bills.
4. **Deterministic Stratified Splits:**
   * Using length-based quantiles, we split the 3,500 pool into:
     * **Train**: 3,000 examples
     * **Validation**: 300 examples
     * **Test**: 200 examples
   * Stratification ensures the validation and test datasets represent the exact same length distributions as the training set, preventing evaluation skew.
   * The split process includes fallback error handling to guarantee reliability if stratification bins become fragile.

### Data Format
Each split is exported as a JSON Lines (`.jsonl`) file in the required instruction-style format:
```json
{
  "instruction": "Summarize the following legal bill.",
  "input": "...",
  "output": "..."
}
```

---

## Setup & Execution

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Data Preparation & Registry Upload
To download, clean, filter, split, and optionally upload the dataset:

1. **(Optional) Configure environment variables for HF dataset registry:**
   ```bash
   export HF_TOKEN="your_hf_write_token"
   export HF_DATASET_REPO="your_username/billsum-processed"
   ```
2. **Execute the script:**
   ```bash
   python3 src/data_prep.py
   ```

This script outputs the processed files to:
* `data/processed/train.jsonl`
* `data/processed/val.jsonl`
* `data/processed/test.jsonl`

If `HF_TOKEN` and `HF_DATASET_REPO` are set, it automatically pushes the processed splits directly to your Hugging Face Datasets repository.

---

## Phase 2: Fine-Tuning with Unsloth (on Kaggle)

We fine-tune **Llama-3.2-3B-Instruct** using QLoRA via Unsloth. The training is intended to run on a **Kaggle Notebook** with dual T4 GPUs (or a single T4 GPU) to utilize the free 30-hour weekly quota.

### Hyperparameter Configuration (Constraint-Driven)
To optimize training speed and conserve GPU hours, we adopt the following standard defaults rather than exhaustive trial-and-error:
* **Base Model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (4-bit quantized version of Llama-3.2 3B).
* **LoRA Configuration:** Rank $r = 16$, alpha $\alpha = 16$, target modules include all major projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Sequence Length Budget:** `max_seq_length = 2048` tokens.
* **Effective Batch Sizing:** Per-device train batch size = `2`, gradient accumulation steps = `4` (simulates a stable effective batch size of `16` on free-tier memory).
* **Optimization:** `adamw_8bit` optimizer, `learning_rate = 2e-4` with linear scheduler, weight decay = `0.01`.
* **Execution Limit:** `max_steps = 60` (~0.2 epochs) for demonstration run, or set to `num_train_epochs = 1` for full coverage, keeping individual runtimes under 15–20 minutes.

### Experiment Tracking & Registry
* **W&B integration:** Runs automatically register to Weights & Biases if `WANDB_API_KEY` is present.
* **HF Model Registry:** Final QLoRA adapter is pushed to Hugging Face Model Hub under the repository specified by `HF_MODEL_REPO`.

### Running on Kaggle

To run this pipeline phase on Kaggle:
1. Create a new Kaggle notebook and enable **GPU T4 x2** accelerator with **Internet Access ON**.
2. Install the optimized Unsloth libraries at the top of the cell:
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unsloth-ai/unsloth.git"
   pip install --no-deps trl peft transformers accelerate bitsandbytes python-dotenv
   ```
3. Upload your `data/processed/` JSONL files into the Kaggle workspace or dataset directory.
4. Set environment secrets in the Kaggle Notebook settings (or using a local `.env` if running on a private GPU):
   * `HF_TOKEN`: Hugging Face write token.
   * `HF_MODEL_REPO`: Destination model repository (e.g. `yourname/billsum-llama3-lora`).
   * `WANDB_API_KEY`: Weights & Biases API key.
5. Run the training script:
   ```bash
   python3 src/train.py
   ```

## Findings

### Gemini Judge (LLM-as-a-Judge) Results
- **Mean Relevance:** 3.42 / 5
- **Mean Factual:** 3.50 / 5
- **Mean Fluency:** 3.92 / 5
- **Overall Average:** 3.61 / 5
- **Number of evaluated examples:** 12

### Observed Failure Patterns
> **Repetition Loops** – The model frequently repeats long legal clauses verbatim (e.g., “Requires the permittee to comply with all applicable Federal and State laws…”) across different examples, indicating limited exposure during fine‑tuning.
> **Truncation at Token Limit** – Several generations end abruptly mid‑sentence, suggesting the `max_new_tokens` setting is hitting its ceiling.

### ROUGE Scores (Baseline)
| Metric | Score |
|--------|-------|
| ROUGE‑1 | 0.490 |
| ROUGE‑2 | 0.301 |
| ROUGE‑L | 0.380 |

