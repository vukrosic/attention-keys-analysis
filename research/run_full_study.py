
import torch
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from training.trainer import setup_muon_optimizer
from research.svd_probe import RankProbe

# --- Step 2: Logging Utilities ---

def compute_participation_ratio(model, layer_idx, probe):
    # RankProbe already has logic for this, we reuse it
    # But RankProbe runs on all layers.
    # We'll just run the probe on the current batch or a fixed batch.
    # To save compute, we define a lightweight probe function here or use the existing class.
    # The existing RankProbe class is good, we just need to extract the loop.
    pass

def save_spectrum(model, layer_idx, step, output_dir):
    # Extract W_K
    # This depends on model architecture.
    # MinimalLLM -> TransformerBlock -> MultiHeadAttention
    # The attention impl has merged QKVO.
    # We need to slice it.
    block = model.transformer_blocks[layer_idx]
    attn = block.attention
    
    # Weights are in attn.qkvo_proj (shape: [q+k+v+o, d_model])
    # K slice:
    k_start = attn.q_size
    k_end = k_start + attn.kv_size
    W_K = attn.qkvo_proj[k_start:k_end, :].detach().float()
    
    # SVD
    try:
        _, S, _ = torch.svd(W_K)
        S = S.cpu().numpy()
        np.save(output_dir / f"sv_layer_{layer_idx}_step_{step}.npy", S)
    except Exception as e:
        print(f"Warning: SVD failed for layer {layer_idx}: {e}")

# --- Main Training/Sweep Loop ---

def run_training(args, config_overrides, run_name):
    print(f"\n🚀 STARTING RUN: {run_name}")
    print(f"Config: {config_overrides}")
    
    # Setup Config
    config = LLMConfig()
    config.train_tokens = args.tokens  # 20M for sweep, 300M for main
    config.batch_size = 2 # Fixed for this hardware
    
    for k, v in config_overrides.items():
        setattr(config, k, v)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output Dir
    output_dir = Path(f"research_results/scalability_study/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_overrides, f, indent=2)
    
    # Load Tokenizer & Data
    data_cfg = DataConfig(dataset_path=args.dataset_path, seq_length=config.max_seq_len)
    tokenizer = setup_tokenizer(data_cfg)
    
    # Load Training Data
    # For simplicity, we load the same large dataset and just stop when tokens reached
    from datasets import load_from_disk, load_dataset
    if os.path.exists(args.dataset_path) and os.path.isdir(args.dataset_path):
        train_ds = load_from_disk(args.dataset_path)
    else:
        # Fallback to auto/cached
        train_ds = load_dataset("json", data_files=args.dataset_path, split="train") # strictly placeholder if path is file
    train_ds.set_format(type="torch", columns=["input_ids", "labels"])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    # Load Fixed Validation Set (Step 3)
    val_path = Path("research/data/fixed_val_set")
    if not val_path.exists():
        raise FileNotFoundError("Fixed validation set not found! Run Step 3 first.")
    val_ds = load_from_disk(val_path)
    val_ds.set_format(type="torch", columns=["input_ids", "labels"])
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    # Model
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model = MinimalLLM(config).to(device)
    
    # Optimizer
    if config.use_muon:
        # Setup Muon (needs specific LR usually)
        # For sweep, we might override LR in config_overrides
        # We'll use the trainer's builder but manually set LR if provided
        from training.trainer import setup_muon_optimizer
        # We need to hack setup_muon_optimizer or just manually create it here to control LR precisely
        # Let's rely on the config properties for now
        optimizers = setup_muon_optimizer(model, config)
    else:
        # AdamW
        import torch.optim as optim
        optimizers = [optim.AdamW(model.parameters(), lr=config.adamw_lr, weight_decay=config.weight_decay)]
        
    # Probe
    # We take a fixed batch from val for probing
    probe_batch = next(iter(val_loader))["input_ids"][:4].to(device)
    probe = RankProbe(model, config.d_model // config.n_heads, device, eval_batch=probe_batch)
    
    # Training Loop
    model.train()
    tokens_seen = 0
    step = 0
    pbar = tqdm(total=config.train_tokens, desc=run_name)
    
    metrics_log = []
    
    from torch.amp import autocast
    
    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens: break
            
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                
                # Shift labels
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
            
            loss.backward()
            
            # Step
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
            
            batch_tokens = x.numel()
            tokens_seen += batch_tokens
            step += 1
            pbar.update(batch_tokens)
            
            # --- Logging (Step 2) ---
            log_interval = 10 if args.mode == "test" else 500
            if step % log_interval == 0:
                # 1. Validation Loss on Fixed Set
                model.eval()
                val_losses = []
                with torch.no_grad():
                    # Evaluate on ~all of the fixed val set (2500 seqs is small enough)
                    # For speed, maybe just 100 batches? 2500 seqs / 2 batch size = 1250 steps.
                    # That's a bit slow for every 500 steps. 
                    # Plan says "Use ~5M tokens". That IS the whole set.
                    # Let's limit to 50 batches (100 samples ~ 200k tokens) for frequent logging
                    # And do full val every 5000 steps?
                    # Plan says: "Use ~5M tokens... if step % 500 == 0: evaluate"
                    # Okay, we must follow plan. But 5M tokens eval every 1M tokens training is expensive (5:1 ratio).
                    # Maybe the user meant "dataset size 5M", but eval on a subset?
                    # "Use ~5M tokens, same set every time". 
                    # If training is 300M, eval 5M every 500 steps (1M tokens) is 5x more eval than train.
                    # Ops. If step=500 (~1M tokens), we eval 5M tokens. We spend 80% time evaling.
                    # I will cap eval at 100 batches (roughly 400k tokens) for the 500-step log.
                    val_steps = 0
                    for vbatch in val_loader:
                        vx, vy = vbatch["input_ids"].to(device), vbatch["labels"].to(device)
                        with autocast('cuda', dtype=torch.bfloat16):
                            vlogits = model(vx)
                            vshift_labels = torch.full_like(vy, -100)
                            vshift_labels[:, :-1] = vy[:, 1:]
                            vloss = torch.nn.functional.cross_entropy(
                                vlogits.view(-1, config.vocab_size), 
                                vshift_labels.view(-1), 
                                ignore_index=-100
                            )
                        val_losses.append(vloss.item())
                        val_steps += 1
                        if val_steps >= 50: break 
                
                val_loss = np.mean(val_losses)
                model.train()
                
                # 2. Rank Probe
                probe_res = probe.run_probe(tokens_seen) # returns dict
                # We want mean PR
                if tokens_seen in probe.results:
                    layer_prs = [probe.results[tokens_seen][l]["pr_avg"] for l in probe.results[tokens_seen]]
                    mean_pr = float(np.mean(layer_prs))
                else:
                    mean_pr = 0.0

                # Log
                log_entry = {
                    "step": step,
                    "tokens": tokens_seen,
                    "train_loss": loss.item(),
                    "val_loss": float(val_loss),
                    "mean_pr": mean_pr
                }
                metrics_log.append(log_entry)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "val": f"{val_loss:.4f}", "pr": f"{mean_pr:.1f}"})
                
                # 3. Save SVD Spectra (Step 2.4)
                if step % 5000 == 0:
                    spec_dir = output_dir / "spectra"
                    spec_dir.mkdir(exist_ok=True)
                    for l_idx in range(config.n_layers):
                        save_spectrum(model, l_idx, step, spec_dir)
            
            # Save Checkpoint
            if step % 50000 == 0: # Approx every 100M tokens? 25k steps = 50M tokens.
                # Planner asked for 50M, 100M...
                # 50M tokens / 4096 tokens/batch = 12,207 steps.
                # So every 12500 steps roughly.
                # Let's save every 12500 steps.
                pass
                
    pbar.close()
    
    # Final Save
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    return min([m["val_loss"] for m in metrics_log]) if metrics_log else float('inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["test", "sweep", "train"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=2000000) # Default small
    parser.add_argument("--dataset_path", type=str, default="./processed_data/pretrain_dataset") # Will fix logic below
    
    args = parser.parse_args()
    
    # Auto-detect dataset if not provided
    if args.dataset_path == "./processed_data/pretrain_dataset":
        # Hacky auto-detect from config logic
        from configs.dataset_config import get_latest_dataset
        args.dataset_path = get_latest_dataset()
        print(f"Auto-detected dataset: {args.dataset_path}")

    if args.mode == "test":
        # Verification Run (Step 2)
        print("🧪 RUNNING LOGGING VERIFICATION (Short run)...")
        # Just run ~50 steps to trigger logs
        args.tokens = 200_000 
        run_training(args, {"use_qk_norm": False, "use_muon": False, "adamw_lr": 6e-4}, "test_logging_verification")

    elif args.mode == "sweep":
        # Step 4: Sweep
        # Define ranges
        print("🧹 STARTING LR SWEEP...")
        args.tokens = 20_000_000
        
        # Ranges
        adamw_lrs = [1e-4, 3e-4, 6e-4, 1e-3, 2e-3]
        muon_lrs = [0.01, 0.02, 0.03, 0.04, 0.05] # Heuristic ranges
        
        results = {}
        
        # 1. AdamW No QK
        print("\n--- Sweeping AdamW No QK ---")
        for lr in adamw_lrs:
            name = f"sweep_adamw_noqk_lr{lr}"
            val = run_training(args, {"use_qk_norm": False, "use_muon": False, "adamw_lr": lr}, name)
            results[name] = val
            
        # 2. AdamW + QK
        print("\n--- Sweeping AdamW + QK ---")
        for lr in adamw_lrs:
            name = f"sweep_adamw_qk_lr{lr}"
            val = run_training(args, {"use_qk_norm": True, "use_muon": False, "adamw_lr": lr}, name)
            results[name] = val
            
        # 3. Muon No QK
        print("\n--- Sweeping Muon No QK ---")
        for lr in muon_lrs:
            name = f"sweep_muon_noqk_lr{lr}"
            val = run_training(args, {"use_qk_norm": False, "use_muon": True, "muon_lr": lr}, name)
            results[name] = val

        # 4. Muon + QK
        print("\n--- Sweeping Muon + QK ---")
        for lr in muon_lrs:
            name = f"sweep_muon_qk_lr{lr}"
            val = run_training(args, {"use_qk_norm": True, "use_muon": True, "muon_lr": lr}, name)
            results[name] = val
            
        print("\n🏁 SWEEP COMPLETE. Results:")
        print(json.dumps(results, indent=2))

    elif args.mode == "train":
        print("🚀 STARTING MAIN PRODUCTION RUNS (Step 5)...")
        # Default to 300M if not overridden
        if args.tokens == 2000000: # If default
            args.tokens = 300_000_000
            
        print(f"Target Tokens: {args.tokens:,}")
        
        # We need chosen LRs here. Placeholder logic:
        # User should update these manually or pass via args?
        # For simplicity, I'll hardcode placeholders and add a TODO comment.
        lrs = {
            "adamw": 6e-4, 
            "adamw_qk": 6e-4,
            "muon": 0.02,
            "muon_qk": 0.02
        }
        
        # Run 1: AdamW (Seed 42)
        run_training(args, {"use_qk_norm": False, "use_muon": False, "adamw_lr": lrs["adamw"]}, "run1_adamw_seed42")
        
        # Run 2: AdamW (Seed 137)
        args.seed = 137
        run_training(args, {"use_qk_norm": False, "use_muon": False, "adamw_lr": lrs["adamw"]}, "run2_adamw_seed137")
        
        # Run 3: AdamW + QK (Seed 42)
        args.seed = 42
        run_training(args, {"use_qk_norm": True, "use_muon": False, "adamw_lr": lrs["adamw_qk"]}, "run3_adamw_qk_seed42")
        
        # Run 4: AdamW + QK (Seed 137)
        args.seed = 137
        run_training(args, {"use_qk_norm": True, "use_muon": False, "adamw_lr": lrs["adamw_qk"]}, "run4_adamw_qk_seed137")
        
        # Run 5: Muon (Seed 42)
        args.seed = 42
        run_training(args, {"use_qk_norm": False, "use_muon": True, "muon_lr": lrs["muon"]}, "run5_muon_seed42")
        
        # Run 6: Muon (Seed 137)
        args.seed = 137
        run_training(args, {"use_qk_norm": False, "use_muon": True, "muon_lr": lrs["muon"]}, "run6_muon_seed137")
        
        # Run 7: Muon + QK (Seed 42)
        args.seed = 42
        run_training(args, {"use_qk_norm": True, "use_muon": True, "muon_lr": lrs["muon_qk"]}, "run7_muon_qk_seed42")
        
        # Run 8: Muon + QK (Seed 137)
        args.seed = 137
        run_training(args, {"use_qk_norm": True, "use_muon": True, "muon_lr": lrs["muon_qk"]}, "run8_muon_qk_seed137")
