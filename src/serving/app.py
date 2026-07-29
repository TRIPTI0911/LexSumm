import time
from threading import Lock

from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel, Field

from src.monitoring.log_to_db import log_inference

MODEL_REPO = "Tripti0911/lexsumm-llama3.2-3b-gguf"
MODEL_FILENAME = "Llama-3.2-3B-Instruct.Q4_K_M.gguf"
DEFAULT_MAX_TOKENS = 256

app = FastAPI(title="LexSumm Serving API")

llm = None
llm_lock = Lock()


def get_llm() -> Llama:
    global llm
    if llm is None:
        with llm_lock:
            if llm is None:
                llm = Llama.from_pretrained(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILENAME,
                    n_ctx=4096,
                    n_threads=2,
                    verbose=False,
                )
    return llm


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_new_tokens: int | None = Field(default=None, ge=32, le=1024)


class SummarizeResponse(BaseModel):
    summary: str


def build_prompt(text: str) -> str:
    # This matches the Llama-3.2 instruction format used during fine-tuning.
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Summarize the following legal bill.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_repo": MODEL_REPO,
        "model_filename": MODEL_FILENAME,
    }


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    started_at = time.perf_counter()
    prompt = build_prompt(request.text.strip())
    max_tokens = request.max_new_tokens or DEFAULT_MAX_TOKENS

    try:
        output = get_llm()(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            stop=["<|eot_id|>"],
            echo=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Inference failed: {exc}") from exc

    summary = output["choices"][0]["text"].strip()
    latency_ms = (time.perf_counter() - started_at) * 1000

    log_inference(
        input_text=request.text,
        output_summary=summary,
        latency_ms=latency_ms,
        model_version=f"{MODEL_REPO}/{MODEL_FILENAME}",
        model_repo=MODEL_REPO,
        model_filename=MODEL_FILENAME,
        status="success",
    )

    return SummarizeResponse(summary=summary)
