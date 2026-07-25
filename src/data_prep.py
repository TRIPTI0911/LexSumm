import os
import json
import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

def clean_text(text):
    """
    Cleans bill text by removing common boilerplate, replacing LaTeX/TeX-style
    quotes, and normalizing whitespace.
    """
    if not text:
        return ""
    
    # Replace TeX-style double quotes `` and '' with standard quotes "
    text = text.replace("``", '"').replace("''", '"')
    
    # Remove recurring header boilerplate in US bills (applied to the first 400 characters only to prevent over-matching)
    # e.g., "108th CONGRESS 1st Session H. R. 1" or similar patterns
    prefix = text[:400]
    cleaned_prefix = re.sub(r'^\s*\d+[a-z]{2}\s+CONGRESS.*?(?:Session|SESSION)\s+\S+\s+', '', prefix, flags=re.IGNORECASE | re.DOTALL)
    if len(cleaned_prefix) < len(prefix):
        text = cleaned_prefix + text[400:]
    
    # Remove section numbers or line numbers that might have been OCR'd or formatted weirdly
    # e.g. "SECTION 1.", "[Page H1234]"
    text = re.sub(r'\[Page\s+[A-Z0-9]+\]', '', text)
    
    # Normalize multiple whitespaces, tabs, and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def get_near_dedup_key(text):
    """
    Normalizes the prefix (first ~800 chars) of a bill text to generate a key 
    for near-duplicate checking. This detects identical bills re-introduced 
    across congressional sessions with minor edits.
    """
    prefix = text[:800]
    # Remove all non-alphanumeric characters and lowercase the string
    normalized = re.sub(r'[^a-z0-9]', '', prefix.lower())
    return normalized[:500]

def is_valid_example(source, summary):
    """
    Quality gate: drops examples with empty summaries, or where summary is longer than source text.
    
    Note on Extractive Shortcuts: We rely on BillSum summaries being professionally written abstracts.
    In cases with scraped summaries, we would check the overlap (e.g. Jaccard similarity or ROUGE overlap)
    between the source and summary to prevent the model from learning copy-paste shortcuts.
    """
    if not source or not summary:
        return False
    if len(summary) >= len(source):
        return False
    return True

def filter_by_length(df, min_src=1500, max_src=6000, min_sum=200, max_sum=1500):
    """
    Filters examples based on character length budgets.
    Calculated to fit comfortably within a 2048 token limit (approx. 8000 characters total).
    """
    initial_len = len(df)
    
    # Apply character length filters
    df = df[
        (df['text'].str.len() >= min_src) & 
        (df['text'].str.len() <= max_src) &
        (df['summary'].str.len() >= min_sum) &
        (df['summary'].str.len() <= max_sum)
    ]
    
    print(f"Length filter ({min_src}-{max_src} chars source, {min_sum}-{max_sum} chars summary): "
          f"kept {len(df)}/{initial_len} examples ({len(df)/initial_len:.1%})")
    return df

def deliberate_subsample(df, target_size):
    """
    Performs a deliberate, deterministic, stratified subsample based on text length.
    This ensures that the final dataset represents a balanced distribution of short,
    medium, and long documents within our context window budget.
    """
    if len(df) <= target_size:
        return df
    
    # Sort by text length
    df = df.copy()
    df['text_len'] = df['text'].str.len()
    df = df.sort_values(by='text_len').reset_index(drop=True)
    
    # Select target_size examples systematically across the sorted distribution
    indices = [int(i * (len(df) - 1) / (target_size - 1)) for i in range(target_size)]
    subsampled_df = df.iloc[indices].copy()
    
    # Drop the temporary length column
    subsampled_df = subsampled_df.drop(columns=['text_len'])
    return subsampled_df

def format_instruction(row):
    """
    Formats the sample into the required instruction-style pair for fine-tuning.
    """
    return {
        "instruction": "Summarize the following legal bill.",
        "input": row['text'],
        "output": row['summary']
    }

def main():
    print("Step 1: Downloading BillSum dataset from Hugging Face...")
    # Load the official splits using the namespace repository name to avoid URI issues
    raw_dataset = load_dataset("FiscalNote/billsum")
    
    # Combine train and test into a single pool to perform uniform cleaning and deliberate splitting
    train_df = pd.DataFrame(raw_dataset['train'])
    test_df = pd.DataFrame(raw_dataset['test'])
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Loaded {len(combined_df)} total raw examples.")

    print("\nStep 2: Cleaning text and removing boilerplate...")
    combined_df['text'] = combined_df['text'].apply(clean_text)
    combined_df['summary'] = combined_df['summary'].apply(clean_text)

    print("\nStep 3: Deduplicating dataset (including near-duplicates)...")
    before_dedup = len(combined_df)
    # Deduplicate based on a normalized key derived from the first 500 alphanumeric characters
    combined_df['near_dedup_key'] = combined_df['text'].apply(get_near_dedup_key)
    combined_df = combined_df.drop_duplicates(subset=['near_dedup_key']).reset_index(drop=True)
    combined_df = combined_df.drop(columns=['near_dedup_key'])
    print(f"Deduplication: removed {before_dedup - len(combined_df)} duplicates/near-duplicates. Remaining: {len(combined_df)}")

    print("\nStep 4: Quality filtering (dropping invalid/empty/flipped examples)...")
    valid_mask = combined_df.apply(lambda row: is_valid_example(row['text'], row['summary']), axis=1)
    combined_df = combined_df[valid_mask].reset_index(drop=True)
    print(f"Quality filter: remaining examples: {len(combined_df)}")

    print("\nStep 5: Applying length constraints (to fit max_seq_length budget)...")
    filtered_df = filter_by_length(combined_df)

    print("\nStep 6: Creating deliberate splits (3000 Train, 300 Val, 200 Test)...")
    # Total required = 3500
    total_needed = 3500
    if len(filtered_df) < total_needed:
        raise ValueError(f"Not enough examples matching criteria. Needed {total_needed}, got {len(filtered_df)}")
    
    # Subsample 3500 examples deliberately to maintain balanced length distribution
    sampled_pool = deliberate_subsample(filtered_df, total_needed)
    
    # Deterministic split:
    # First, separate Test (200) from Train + Val (3300)
    # Use stratified split on length deciles or stable deterministic split
    sampled_pool['len_bucket'] = pd.qcut(sampled_pool['text'].str.len(), q=10, labels=False, duplicates='drop')
    
    try:
        train_val_df, test_df = train_test_split(
            sampled_pool, 
            test_size=200, 
            random_state=42, 
            stratify=sampled_pool['len_bucket']
        )
    except ValueError:
        print("Warning: Stratified train/test split failed. Falling back to non-stratified split.")
        train_val_df, test_df = train_test_split(
            sampled_pool, 
            test_size=200, 
            random_state=42, 
            stratify=None
        )
    
    # Next, split Train (3000) and Val (300)
    try:
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=300, 
            random_state=42, 
            stratify=train_val_df['len_bucket']
        )
    except ValueError:
        print("Warning: Stratified train/val split failed. Falling back to non-stratified split.")
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=300, 
            random_state=42, 
            stratify=None
        )
    
    # Clean up temp bucket column
    train_df = train_df.drop(columns=['len_bucket'])
    val_df = val_df.drop(columns=['len_bucket'])
    test_df = test_df.drop(columns=['len_bucket'])

    print(f"Final sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Display some stats
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        avg_src = df['text'].str.len().mean()
        avg_sum = df['summary'].str.len().mean()
        print(f"  {name} set - Avg source length: {avg_src:.1f} chars, Avg summary length: {avg_sum:.1f} chars")

    print("\nStep 7: Formatting to instruction pairs and saving...")
    os.makedirs("data/processed", exist_ok=True)
    
    for name, df, path in [
        ("Train", train_df, "data/processed/train.jsonl"),
        ("Val", val_df, "data/processed/val.jsonl"),
        ("Test", test_df, "data/processed/test.jsonl")
    ]:
        formatted_records = [format_instruction(row) for _, row in df.iterrows()]
        with open(path, "w", encoding="utf-8") as f:
            for record in formatted_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(formatted_records)} records to {path}")

    # Optional Step 8: Push to Hugging Face Hub (Data Registry)
    hf_token = os.getenv("HF_TOKEN")
    hf_dataset_repo = os.getenv("HF_DATASET_REPO")
    if hf_token and hf_dataset_repo:
        print(f"\nStep 8: Pushing processed splits to Hugging Face Hub dataset registry ({hf_dataset_repo})...")
        try:
            from datasets import Dataset, DatasetDict
            
            # Read jsonl files back to create a standard Dataset Dict
            dataset_dict = DatasetDict({
                "train": Dataset.from_json("data/processed/train.jsonl"),
                "validation": Dataset.from_json("data/processed/val.jsonl"),
                "test": Dataset.from_json("data/processed/test.jsonl")
            })
            
            dataset_dict.push_to_hub(hf_dataset_repo, token=hf_token)
            print("Successfully pushed dataset to Hugging Face Hub dataset registry!")
        except Exception as e:
            print(f"Failed to push dataset to Hugging Face Hub: {e}")
    else:
        print("\nStep 8: HF Dataset Registry upload skipped. (Set HF_TOKEN and HF_DATASET_REPO env variables to enable.)")

    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
