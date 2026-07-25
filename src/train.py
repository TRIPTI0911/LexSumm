import os
import sys
import re
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

# Load local environment variables if available (.env)
load_dotenv()

def formatting_prompts_func(examples):
    """
    Formats raw instruction-input-output pairs into Llama-3 instruction template.
    """
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Format matching Llama-3/3.2 chat template:
        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )
        texts.append(text)
    return {"text": texts}

def main():
    print("=== LLMOps Fine-Tuning Pipeline: Phase 2 (QLoRA via Unsloth) ===")

    # 1. Environment and GPU Validation
    if not torch.cuda.is_available():
        print("\n[-] GPU not detected. Unsloth requires a GPU-accelerated environment.")
        print("[-] Skipping training run. This script is fully prepared for Kaggle/Colab GPU runtimes.")
        sys.exit(0)

    print(f"\n[+] GPU detected: {torch.cuda.get_device_name(0)}")
    
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("\n[-] Unsloth/TRL/Transformers libraries not found.")
        print("[-] To install Unsloth (specifically for Kaggle/Colab environments with CUDA 12.1+):")
        print("    pip install \"unsloth[colab-new] @ git+https://github.com/unsloth-ai/unsloth.git\"")
        print("    pip install --no-deps trl peft transformers accelerate bitsandbytes")
        sys.exit(1)

    # 2. Hyperparameters & Settings
    max_seq_length = 2048
    dtype = None # Auto-detected (Float16/Bfloat16 depending on GPU)
    load_in_4bit = True
    base_model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

    # Load paths
    train_path = os.getenv("TRAIN_DATA_PATH", "data/processed/train.jsonl")
    val_path = os.getenv("VAL_DATA_PATH", "data/processed/val.jsonl")

    # 3. Model Loading & PEFT Configuration
    print(f"\n[+] Loading base model: {base_model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    print("\n[+] Injecting LoRA adapters (QLoRA)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0, # Optimized by Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth", # 70%+ memory savings
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 4. Dataset Loading and Formatting
    print("\n[+] Loading datasets...")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[+] Local files found. Loading from {train_path} and {val_path}...")
        train_dataset = Dataset.from_json(train_path)
        val_dataset = Dataset.from_json(val_path)
    else:
        hf_dataset_repo_data = os.getenv("HF_DATASET_REPO", "Tripti0911/billsum-processed")
        # Normalize the repo name to remove any domain/prefix if user copied it from web URL
        hf_dataset_repo_data = re.sub(r'^(?:https?://)?(?:huggingface\.co/datasets/)?', '', hf_dataset_repo_data)
        print(f"[+] Local files not found. Loading from HF Hub dataset registry: {hf_dataset_repo_data}...")
        try:
            train_dataset = load_dataset(hf_dataset_repo_data, split="train")
            val_dataset = load_dataset(hf_dataset_repo_data, split="validation")
        except Exception as e:
            print(f"\n[-] Failed to load dataset from Hugging Face Hub: {e}")
            print("[-] Please run 'python3 src/data_prep.py' locally or set a valid HF_DATASET_REPO environment variable.")
            sys.exit(1)

    print("[+] Formatting data into Llama-3 instruction templates...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    # 5. Weights & Biases Logging Integration
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        print("\n[+] W&B API Key detected. Initializing experiment tracking...")
        import wandb
        wandb.login(key=wandb_key)
        report_to = "wandb"
    else:
        print("\n[-] W&B API Key not set. Training run will not be logged online.")
        report_to = "none"

    # 6. SFT Trainer Instantiation
    print("\n[+] Configuring SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4, # Effective batch size = 2 * 2 * 4 = 16
            warmup_steps=5,
            max_steps=60, # Keep steps low to conserve free Kaggle quota (30 hrs/week)
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=20,
            report_to=report_to,
        ),
    )

    # 7. Start Training
    print("\n[+] Beginning training...")
    trainer_stats = trainer.train()
    print("\n[+] Training complete!")
    print(f"    Peak reserved memory: {torch.cuda.max_memory_reserved() / 1e9} GB")
    print(f"    Total training time: {trainer_stats.metrics['train_runtime']} seconds")

    # 8. Save Model Adapters Locally
    output_dir = "lora_adapter"
    print(f"\n[+] Saving LoRA adapter locally to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 9. Push to Hugging Face Model Registry (Optional)
    hf_token = os.getenv("HF_TOKEN")
    hf_model_repo = os.getenv("HF_MODEL_REPO")
    if hf_token and hf_model_repo:
        print(f"\n[+] Pushing LoRA adapter to Hugging Face Model Registry ({hf_model_repo})...")
        try:
            model.push_to_hub(hf_model_repo, token=hf_token)
            tokenizer.push_to_hub(hf_model_repo, token=hf_token)
            print("[+] Successfully pushed LoRA adapter to HF Hub!")
        except Exception as e:
            print(f"[-] Failed to push model to Hugging Face Hub: {e}")
    else:
        print("\n[-] Hugging Face registry upload skipped. (Set HF_TOKEN and HF_MODEL_REPO to enable.)")

    print("\nPipeline Phase 2 finished successfully!")

if __name__ == "__main__":
    main()
