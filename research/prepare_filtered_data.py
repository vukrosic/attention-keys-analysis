import os
import sys
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def estimate_complexity(text):
    """
    Very simple heuristic for complexity:
    - Average word length
    - Presence of technical keywords
    """
    words = text.split()
    if not words: return 0
    avg_len = sum(len(w) for w in words) / len(words)
    
    # Simple keyword list for "complexity"
    technical_keywords = ["theorem", "equation", "analysis", "synthesis", "quantum", "biological", "infrastructure", "algorithm", "hypothesis"]
    keyword_count = sum(1 for w in words if w.lower() in technical_keywords)
    
    return avg_len + (keyword_count * 2.0)

def prepare_filtered_cosmo(target_tokens, subset_type="simple", output_dir="./processed_data"):
    print(f"🚀 Filtering Cosmopedia for {subset_type} subset ({target_tokens:,} tokens)...")
    
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    output_path = os.path.join(output_dir, f"cosmo_{subset_type}_{target_tokens}")
    os.makedirs(output_dir, exist_ok=True)
    
    iterator = iter(ds)
    current_tokens = 0
    buffer = []
    
    pbar = tqdm(total=target_tokens)
    
    while current_tokens < target_tokens:
        try:
            item = next(iterator)
            text = item['text']
            
            score = estimate_complexity(text)
            
            # Simple heuristic thresholds
            if subset_type == "simple" and score > 5.5: continue
            if subset_type == "complex" and score < 7.0: continue
            
            ids = tokenizer.encode(text, add_special_tokens=True)
            # Only use full sequences (2048) or chunk them?
            # For simplicity, let's just collect tokens and chunk later like prepare_mix_data.py
            buffer.append({"input_ids": ids})
            current_tokens += len(ids)
            pbar.update(len(ids))
            
        except StopIteration:
            break
            
    pbar.close()
    
    # Save to disk as a HF dataset
    print(f"Saving to {output_path}...")
    final_ds = Dataset.from_list(buffer)
    final_ds.save_to_disk(output_path)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=25_000_000)
    parser.add_argument("--type", type=str, choices=["simple", "complex"], required=True)
    args = parser.parse_args()
    
    prepare_filtered_cosmo(args.tokens, args.type)
