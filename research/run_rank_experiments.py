import torch
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from configs.dataset_config import DataConfig
from data.loader import setup_tokenizer
from train_llm import prepare_datasets
from training.trainer import setup_muon_optimizer, train_model
from research.svd_probe import RankProbe

def run_experiment(name, config_overrides, dataset_path="auto", target_tokens=25_000_000):
    print(f"\n" + "="*80)
    print(f"🚀 RUNNING EXPERIMENT: {name}")
    print("="*80)
    
    config = LLMConfig()
    config.train_tokens = target_tokens
    config.compile_model = False # Disable for now to ensure we see logs immediately
    config.batch_size = 2 # Reduced for 24GB VRAM with 2048 seq_len
    
    # Apply overrides
    for k, v in config_overrides.items():
        setattr(config, k, v)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup data
    print(f"Loading datasets from {dataset_path}...")
    data_cfg = DataConfig(dataset_path=dataset_path, seq_length=config.max_seq_len)
    tokenizer = setup_tokenizer(data_cfg)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    # Initialize model
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.empty_cache()
    model = MinimalLLM(config).to(device)
    
    # Prepare fixed eval batch for SVD
    # We take the first few samples from val_loader
    eval_batch_iter = iter(val_loader)
    eval_batch = next(eval_batch_iter)["input_ids"][:4] # 4 samples * 2048 tokens = 8k tokens
    
    # Setup Probe
    probe = RankProbe(model, config.d_model // config.n_heads, device, eval_batch=eval_batch)
    
    # Setup Optimizer
    optimizers = setup_muon_optimizer(model, config)
    
    # Tracking ranks at specific milestones
    milestones = [0, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 15_000_000, 25_000_000]
    
    # Modified training loop to include probing
    results_dir = Path(f"research_results/rank_study/{name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # We'll use a wrapper or inline training for simplicity since we want custom probing
    # Actually, let's just use train_model and pass a callback if it supported it.
    # It doesn't, so let's implement a minimal version here or modify train_model.
    # Better to copy-paste the core loop and add the probe call.
    
    print(f"Training for {config.train_tokens:,} tokens...")
    model.train()
    tokens_seen = 0
    step = 0
    pbar = tqdm(total=config.train_tokens)
    
    metrics_log = []
    
    # Initial probe
    probe.run_probe(0)
    
    # Mixed precision setup
    from torch.amp import autocast
    
    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens: break
            
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            
            # Forward + Backward with autocast
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                # Shift labels for CE loss
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
            
            # Milestone probing
            for m in milestones:
                if tokens_seen >= m and tokens_seen - batch_tokens < m:
                    print(f"\n[Milestone {m}] Running SVD Probe...")
                    probe.run_probe(tokens_seen)
                    
                    # Save intermediate results
                    with open(results_dir / "probe_results.json", "w") as f:
                        json.dump(probe.results, f, indent=2)
                        
            if step % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                metrics_log.append({"step": step, "tokens": tokens_seen, "loss": loss.item()})

    pbar.close()
    
    # Final save
    with open(results_dir / "training_log.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    torch.save(model.state_dict(), results_dir / "model_final.pt")
    print(f"✅ Experiment {name} complete.")

if __name__ == "__main__":
    # The 6 Planned Experiments
    experiments = [
        ("A1_Baseline_QK_Muon", {"use_qk_norm": True, "use_muon": True}, "auto"),
        ("A2_NoQK_Muon", {"use_qk_norm": False, "use_muon": True}, "auto"),
        ("B1_Baseline_QK_AdamW", {"use_qk_norm": True, "use_muon": False}, "auto"),
        ("B2_NoQK_AdamW", {"use_qk_norm": False, "use_muon": False}, "auto"),
        ("C1_SimpleData_QK_Muon", {"use_qk_norm": True, "use_muon": True}, "./processed_data/cosmo_simple_25000000"), 
        ("C2_ComplexData_QK_Muon", {"use_qk_norm": True, "use_muon": True}, "./processed_data/cosmo_complex_25000000"), 
    ]
    
    # Production Run: 25M tokens for AdamW experiments (B1, B2)
    target_tokens = 25_000_000
    adamw_exps = experiments[2:4] # B1, B2
    
    print(f"\n{'='*80}\n🚀 PRODUCTION MODE: Running {target_tokens:,} tokens for AdamW experiments\n{'='*80}")
    
    for name, overrides, path in adamw_exps:
        # Use the specific mix path we just prepared to ensure parity with the Muon run
        run_experiment(name, overrides, dataset_path=path, target_tokens=target_tokens)
    
    print("\n" + "="*80)
    print("✅ ADAMW PRODUCTION EXPERIMENTS COMPLETE.")
    print("="*80)
