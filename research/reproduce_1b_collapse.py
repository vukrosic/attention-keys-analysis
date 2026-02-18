
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config_1b import LLMConfig1B
from training.trainer import setup_muon_optimizer
from research.svd_probe import RankProbe, compute_rank_metrics
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets

def run_reproduction(use_qk_norm, tokens=10000000):
    run_name = f"repro_1b_{'QK' if use_qk_norm else 'NoQK'}"
    print(f"\n🚀 STARTING REPRODUCTION: {run_name} ({tokens:,} tokens)")
    
    config = LLMConfig1B()
    config.use_qk_norm = use_qk_norm
    config.train_tokens = tokens
    config.batch_size = 1
    config.gradient_accumulation_steps = 8
    config.compile_model = False 
    config.gradient_checkpointing = True # Enable for 24GB safety
    
    device = torch.device('cuda')
    
    # Data
    dataset_path = "/root/llm-research-kit/processed_data/pretrain_mix_26000000"
    data_cfg = DataConfig(dataset_path=dataset_path, seq_length=config.max_seq_len)
    tokenizer = setup_tokenizer(data_cfg)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    # Fixed eval batch for PR
    eval_batch = next(iter(DataLoader(val_ds, batch_size=4)))["input_ids"].to(device)
    
    # Model
    torch.manual_seed(42)
    model = MinimalLLM(config).to(device)
    
    # Probe
    # We'll run probe every 500 steps
    probe = RankProbe(model, config.d_model // config.n_heads, device, eval_batch=eval_batch)
    
    # Optimizer
    optimizers = setup_muon_optimizer(model, config)
    
    # Training Loop
    model.train()
    tokens_seen = 0
    step = 0
    pbar = tqdm(total=config.train_tokens, desc=run_name)
    
    results = {
        "tokens": [],
        "val_loss": [],
        "mean_pr": [],
        "layer_pr": {}
    }
    
    output_dir = Path(f"research_results/repro_1b/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens: break
                
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
                loss = loss / config.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                for opt in optimizers: opt.step(); opt.zero_grad()
            
            batch_tokens = x.numel()
            tokens_seen += batch_tokens
            step += 1
            pbar.update(batch_tokens)
            
            if step % 500 == 0:
                # Probe
                pr_dict = probe.run_probe(tokens_seen)
                mean_pr = np.mean([v["pr_avg"] for v in pr_dict.values()])
                
                results["tokens"].append(tokens_seen)
                results["mean_pr"].append(float(mean_pr))
                results["val_loss"].append(float(loss.item() * config.gradient_accumulation_steps))
                
                # Per layer
                for l_idx, l_metrics in pr_dict.items():
                    if l_idx not in results["layer_pr"]: results["layer_pr"][l_idx] = []
                    results["layer_pr"][l_idx].append(l_metrics["pr_avg"])
                
                pbar.set_postfix({"loss": f"{loss.item()*8:.4f}", "pr": f"{mean_pr:.2f}"})
                
                # Partial Save
                with open(output_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2)

    pbar.close()
    return results

if __name__ == "__main__":
    # Run QK
    res_qk = run_reproduction(use_qk_norm=True, tokens=1000000)
    
    # Run NoQK
    # Clear cache
    torch.cuda.empty_cache()
    res_no_qk = run_reproduction(use_qk_norm=False, tokens=1000000)
    
    print("\n✅ Reproduction Complete!")
