import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np
import math
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.getcwd())
from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from training.trainer import setup_muon_optimizer
from torch.utils.data import DataLoader
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets

class CivilWarTracker:
    def __init__(self, model):
        self.stats = {}
        self.hooks = []
        self.model = model
        self.step = 0
        
        # Temporary storage for pairing Q and K
        self._temp_q = {}
        self._temp_k = {}

        for i, block in enumerate(model.transformer_blocks):
            layer_name = f"layer_{i}"
            # Statistics structure: Layer -> Head -> Metric -> List of values
            # We track KV heads specifically since they control the k_norm gain
            num_kv_heads = block.attention.n_kv_heads
            
            self.stats[layer_name] = {
                "head_stats": [ {
                    "l2_norm": [],
                    "max_abs": [],
                    "pr": [],        # Participation Ratio (Effective Rank)
                    "entropy": [],   # Shannon Entropy
                    "qk_ratio": [],  # ||Q|| / ||K||
                } for _ in range(num_kv_heads) ],
                "gamma": [],         # The k_norm gain vector [KV_HEADS, D_K]
                "steps": []
            }

            def make_q_hook(name, l_idx):
                def hook(module, input, output):
                    # output: [B, T, H, D]
                    self._temp_q[l_idx] = output.detach()
                return hook

            def make_k_hook(name, l_idx):
                def hook(module, input, output):
                    # output: [B, T, H_kv, D]
                    q = self._temp_q.get(l_idx)
                    if q is None: return # Should not happen in serial exec
                    
                    k = output.detach()
                    
                    # Compute metrics
                    with torch.no_grad():
                        # Basic norms per head
                        # Mean over Batch and Time
                        h_norms = torch.norm(k, dim=-1).mean(dim=(0, 1)) # [H_kv]
                        h_max = k.abs().max(dim=-1)[0].mean(dim=(0, 1)) # [H_kv]
                        
                        # Participation Ratio: (sum x^2)^2 / sum x^4
                        sq = k.pow(2).sum(dim=-1)
                        quad = k.pow(4).sum(dim=-1)
                        h_pr = (sq.pow(2) / quad).mean(dim=(0, 1)) # [H_kv]
                        
                        # Q/K Ratio
                        # Since Q might have more heads than K (GQA), we repeat K to match Q
                        # Or for simplicity, just compare the norms of corresponding chunks
                        # Let's handle GQA by averaging Q heads that share a K head
                        n_heads = q.shape[2]
                        n_kv_heads = k.shape[2]
                        group_size = n_heads // n_kv_heads
                        
                        q_norms = torch.norm(q, dim=-1).mean(dim=(0, 1)) # [H]
                        # Group average Q norms to match K heads
                        q_norms_grouped = q_norms.view(n_kv_heads, group_size).mean(dim=1)
                        h_qk_ratio = (q_norms_grouped / h_norms)

                        # Entropy
                        # This is expensive. Let's do it for a small sample
                        # Sample 1 batch, 128 tokens
                        B_s, T_s = min(k.shape[0], 1), min(k.shape[1], 128)
                        q_s = q[0:B_s, 0:T_s] # [B, T, H, D]
                        k_s = k[0:B_s, 0:T_s] # [B, T, H_kv, D]
                        
                        # Repeat K for GQA
                        k_s_rep = torch.repeat_interleave(k_s, group_size, dim=2)
                        
                        # Compute attention: [H, T, T]
                        # SDPA scale is 1/sqrt(D)
                        d_k = k.shape[-1]
                        # Transpose for batch matmul: [H, T, D] @ [H, D, T]
                        q_s = q_s.permute(2, 0, 1, 3).reshape(n_heads, -1, d_k)
                        k_s_rep = k_s_rep.permute(2, 0, 1, 3).reshape(n_heads, -1, d_k)
                        
                        scores = torch.bmm(q_s, k_s_rep.transpose(1, 2)) / math.sqrt(d_k)
                        probs = F.softmax(scores, dim=-1) # [H, T, T]
                        
                        # Entropy: -sum p log p
                        # Add epsilon to avoid log(0)
                        en = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean(dim=-1) # [H]
                        # Average entropy across KV groups
                        h_entropy = en.view(n_kv_heads, group_size).mean(dim=1)

                        # Store stats
                        self.stats[name]["steps"].append(self.step)
                        for h in range(n_kv_heads):
                            self.stats[name]["head_stats"][h]["l2_norm"].append(h_norms[h].item())
                            self.stats[name]["head_stats"][h]["max_abs"].append(h_max[h].item())
                            self.stats[name]["head_stats"][h]["pr"].append(h_pr[h].item())
                            self.stats[name]["head_stats"][h]["qk_ratio"].append(h_qk_ratio[h].item())
                            self.stats[name]["head_stats"][h]["entropy"].append(h_entropy[h].item())
                            
                        # Also store the gain weights (gamma)
                        # RMSNorm.weight is [D_k]. Wait, RMSNorm in models/layers.py is per-head dimension.
                        # So it's shared across heads?
                        # Actually, looking at MultiHeadAttention.__init__:
                        # self.k_norm = nn.RMSNorm(self.d_k)
                        # This means one weight vector of size d_k is shared across ALL heads in that layer.
                        # This is an important detail! The "Civil War" happens DESPITE sharing the same gain.
                        self.stats[name]["gamma"].append(module.weight.detach().cpu().numpy().tolist())
                        
                return hook

            q_hook = block.attention.q_norm.register_forward_hook(make_q_hook(layer_name, i))
            k_hook = block.attention.k_norm.register_forward_hook(make_k_hook(layer_name, i))
            self.hooks.extend([q_hook, k_hook])

    def update_step(self, step):
        self.step = step

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def save_stats(self, path):
        with open(path, 'w') as f:
            json.dump(self.stats, f)

def run_experiment():
    print("⚔️ Initializing Civil War Tracking Experiment")
    
    config = LLMConfig()
    config.train_tokens = 4_000_000 # Enough to see divergence
    config.batch_size = 4
    config.compile_model = False
    
    # Setup data
    data_cfg = DataConfig(
        dataset_path="auto",
        seq_length=config.max_seq_len,
        num_samples=10000,
        cache_dir="./hf_cache",
    )
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinimalLLM(config).to(device)
    
    # Tracker
    tracker = CivilWarTracker(model)
    optimizers = setup_muon_optimizer(model, config)
    
    tokens_seen = 0
    step = 0
    pbar = tqdm(total=config.train_tokens)
    model.train()
    
    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens: break
                
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            tracker.update_step(step)
            
            logits = model(x)
            shift_labels = torch.full_like(y, -100)
            shift_labels[:, :-1] = y[:, 1:]
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            
            loss.backward()
            for opt in optimizers: opt.step(); opt.zero_grad()
                
            batch_tokens = x.numel()
            tokens_seen += batch_tokens
            step += 1
            pbar.update(batch_tokens)
            
            if step % 50 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()
    tracker.remove_hooks()
    os.makedirs("research_results", exist_ok=True)
    tracker.save_stats("research_results/civil_war_stats.json")
    print("\n✅ Experiment Complete. Stats saved to research_results/civil_war_stats.json")

if __name__ == "__main__":
    run_experiment()
