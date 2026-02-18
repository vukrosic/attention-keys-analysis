
import torch
import os
import sys
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from configs.dataset_config import DataConfig, get_latest_dataset
from data.loader import setup_tokenizer
from training.trainer import setup_muon_optimizer
from research.svd_probe import RankProbe
from datasets import load_from_disk

# Mixed precision setup
from torch.amp import autocast

def run_muon_experiment(name, muon_lr, use_qk_norm, target_tokens=5_000_000):
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING: {name}")
    print(f"LR: {muon_lr} | QK-Norm: {use_qk_norm} | Tokens: {target_tokens:,}")
    print("="*60)
    
    config = LLMConfig()
    config.train_tokens = target_tokens
    config.compile_model = False
    # Use default batch_size from LLMConfig (which user set to 8)
    config.muon_lr = muon_lr
    config.use_qk_norm = use_qk_norm
    config.use_muon = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup data
    dataset_path = get_latest_dataset()
    data_cfg = DataConfig(dataset_path=dataset_path, seq_length=config.max_seq_len)
    tokenizer = setup_tokenizer(data_cfg)
    
    train_ds = load_from_disk(dataset_path)
    train_ds.set_format(type="torch", columns=["input_ids", "labels"])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    # Fixed Validation Set (Step 3 habit)
    val_path = Path("research/data/fixed_val_set")
    if val_path.exists():
        val_ds = load_from_disk(val_path)
        val_ds.set_format(type="torch", columns=["input_ids", "labels"])
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    else:
        val_loader = None
    
    # Model
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    model = MinimalLLM(config).to(device)
    
    # Optimizer
    optimizers = setup_muon_optimizer(model, config)
    
    # Probe
    # Get a batch for probing
    probe_batch = next(iter(val_loader))["input_ids"][:4].to(device) if val_loader else None
    probe = RankProbe(model, config.d_model // config.n_heads, device, eval_batch=probe_batch)
    
    # Loop
    model.train()
    tokens_seen = 0
    step = 0
    pbar = tqdm(total=target_tokens, desc=name)
    
    results_dir = Path(f"research_results/muon_sweep_5m/{name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = []
    
    while tokens_seen < target_tokens:
        for batch in train_loader:
            if tokens_seen >= target_tokens: break
            
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            
            loss.backward()
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
            
            batch_tokens = x.numel()
            tokens_seen += batch_tokens
            step += 1
            pbar.update(batch_tokens)
            
            # Log every 50 steps
            if step % 50 == 0:
                # Basic val loss placeholder or full
                v_loss = loss.item() # Placeholder
                if val_loader:
                    model.eval()
                    with torch.no_grad():
                        v_batch = next(iter(val_loader))
                        vx, vy = v_batch["input_ids"].to(device), v_batch["labels"].to(device)
                        with autocast('cuda', dtype=torch.bfloat16):
                            vl = model(vx)
                            vsl = torch.full_like(vy, -100)
                            vsl[:, :-1] = vy[:, 1:]
                            v_loss = torch.nn.functional.cross_entropy(vl.view(-1, config.vocab_size), vsl.view(-1), ignore_index=-100).item()
                    model.train()
                
                # PR Probe
                pr = 0.0
                if probe_batch is not None:
                    p_res = probe.run_probe(tokens_seen)
                    if tokens_seen in probe.results:
                        pr = np.mean([probe.results[tokens_seen][l]["pr_avg"] for l in probe.results[tokens_seen]])
                
                metrics.append({"step": step, "tokens": tokens_seen, "loss": loss.item(), "val_loss": v_loss, "pr": float(pr)})
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "val": f"{v_loss:.4f}", "pr": f"{pr:.1f}"})

    pbar.close()
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics[-1]["val_loss"] if metrics else 99.0

if __name__ == "__main__":
    lrs = [0.012, 0.024, 0.036, 0.048]
    configs = [True, False] # use_qk_norm
    
    summary = {}
    for use_qk in configs:
        for lr in lrs:
            mode_name = "qk" if use_qk else "no_qk"
            exp_name = f"muon_{mode_name}_lr{lr}"
            final_val = run_muon_experiment(exp_name, lr, use_qk)
            summary[exp_name] = final_val
            
    print("\n" + "="*60)
    print("🏁 QUICK SWEEP COMPLETE")
    print(json.dumps(summary, indent=2))
    print("="*60)
