
import os
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Add project root to sys.path
sys.path.append(os.getcwd())

from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer

def create_val_set():
    print("🚀 Creating Fixed Validation Set (Step 3)...")
    
    # Config
    # We use the same dataset config as production runs will use
    # "FineWeb-Edu + Cosmopedia Mix" path from run_rank_experiments.py
    # But wait, run_rank_experiments.py uses "./processed_data/cosmo_simple_25000000" or similar.
    # The user manual says "FineWeb-Edu + Cosmopedia Mix".
    # I will try to load from the 'auto' path or the one used in previous experiments.
    # Let's use "./processed_data/cosmo_complex_25000000" as a good source, or just use the HuggingFace dataset directly if needed.
    # Actually, better to use the exact same tokenizer and source.
    
    # Let's use the default "auto" which finds the latest locally processed data.
    # This ensures distribution matching with training data.
    data_cfg = DataConfig(dataset_path="auto", split="train", seq_length=2048)
    
    # However, "val" split might not exist in the local processed data if it was just a raw dump.
    # If "auto" points to a processed corpus, we can just split it.
    
    print(f"Loading dataset from: {data_cfg.dataset_path}")
    if os.path.isdir(data_cfg.dataset_path):
        from datasets import load_from_disk
        try:
            ds = load_from_disk(data_cfg.dataset_path)
        except:
            # Fallback for raw text folder
            ds = load_dataset(data_cfg.dataset_path, split="train")
    else:
         ds = load_dataset(data_cfg.dataset_path, split="train")

    print(f"Dataset loaded. Size: {len(ds)}")
    
    # We need 5M tokens.
    # Seq len = 2048.
    # 5,000,000 / 2048 ~= 2441 sequences.
    num_val_sequences = 2500
    
    # Shuffle with fixed seed for reproducibility
    ds = ds.shuffle(seed=42)
    
    # Select val set
    val_subset = ds.select(range(num_val_sequences))
    
    # Save to disk
    val_path = Path("research/data/fixed_val_set")
    val_path.mkdir(parents=True, exist_ok=True)
    val_subset.save_to_disk(val_path)
    
    print(f"✅ Fixed validation set saved to {val_path}")
    print(f"sequences: {len(val_subset)}")
    print(f"approx tokens: {len(val_subset) * 2048:,}")

if __name__ == "__main__":
    create_val_set()
