import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np

# Add project root to sys.path
sys.path.append(os.getcwd())
from tqdm import tqdm
from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from training.trainer import setup_muon_optimizer
from torch.utils.data import DataLoader
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets

class KeyTracker:
    def __init__(self, model):
        self.stats = {}
        self.hooks = []
        self.model = model
        self.step = 0
        
        for i, block in enumerate(model.transformer_blocks):
            layer_name = f"layer_{i}"
            self.stats[layer_name] = {
                "steps": [], 
                "mean_norm": [], 
                "max_abs": [],
                "std": [],
                "cosine_sim": [] # Average cosine similarity between random keys
            }
            
            # Hook specifically into k_norm to only get Key statistics
            # MultiHeadAttention has k_norm
            def get_hook(name):
                def hook(module, input, output):
                    # output shape: [B, T, H, D]
                    with torch.no_grad():
                        # Basic stats
                        norm = torch.norm(output, dim=-1).mean().item()
                        max_val = output.abs().max().item()
                        std = output.std().item()
                        
                        # Cosine similarity within heads (sample 100 pairs)
                        # output: [B, T, H, D]
                        B, T, H, D = output.shape
                        # Pick a random head and a few tokens
                        h_idx = torch.randint(0, H, (1,)).item()
                        tokens = output[0, :, h_idx, :] # [T, D]
                        # Sample 10 random pairs of tokens
                        if T > 1:
                            idx1 = torch.randint(0, T, (10,))
                            idx2 = torch.randint(0, T, (10,))
                            t1 = tokens[idx1] # [10, D]
                            t2 = tokens[idx2] # [10, D]
                            sim = F.cosine_similarity(t1, t2, dim=-1).mean().item()
                        else:
                            sim = 1.0
                            
                        self.stats[name]["steps"].append(self.step)
                        self.stats[name]["mean_norm"].append(norm)
                        self.stats[name]["max_abs"].append(max_val)
                        self.stats[name]["std"].append(std)
                        self.stats[name]["cosine_sim"].append(sim)
                return hook

            hook_ref = block.attention.k_norm.register_forward_hook(get_hook(layer_name))
            self.hooks.append(hook_ref)

    def update_step(self, step):
        self.step = step

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def save_stats(self, path):
        with open(path, 'w') as f:
            json.dump(self.stats, f)

def run_experiment():
    print("🚀 Starting Advanced Key Vector Analysis")
    
    # Setup config for a quick but meaningful run
    config = LLMConfig()
    config.train_tokens = 8_000_000 # Increased for requested depth
    config.batch_size = 4
    config.compile_model = False # Disable for hooks
    
    # Setup data
    data_cfg = DataConfig(
        dataset_path="auto",
        seq_length=config.max_seq_len,
        num_samples=8000,
        cache_dir="./hf_cache",
    )
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    
    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinimalLLM(config).to(device)
    
    # Hook it up
    tracker = KeyTracker(model)
    
    optimizers = setup_muon_optimizer(model, config)
    
    print(f"Training for {config.train_tokens} tokens...")
    tokens_seen = 0
    step = 0
    
    pbar = tqdm(total=config.train_tokens)
    
    model.train()
    # Use bfloat16 for speed if available
    use_amp = torch.cuda.is_available()
    
    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens:
                break
                
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            
            tracker.update_step(step)
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                logits = model(x)
                
                # Shift labels
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            
            loss.backward()
            
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
                
            batch_tokens = x.numel()
            tokens_seen += batch_tokens
            step += 1
            pbar.update(batch_tokens)
            
            if step % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()
    tracker.remove_hooks()
    
    # Save and Plot
    os.makedirs("research_results", exist_ok=True)
    tracker.save_stats("research_results/key_stats.json")
    
    plot_results(tracker.stats)

def smooth_data(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_results(stats):
    layers = list(stats.keys())
    steps = stats[layers[0]]["steps"]
    
    # Use a modern, cleaner style
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Helper to get smoothed steps
    def get_smooth_steps(s, window=10):
        return s[window-1:]

    # 1. Mean Norm (Comparison of stages)
    ax = axes[0, 0]
    stages = {
        "Early Layers": layers[:2],
        "Middle Layers": layers[len(layers)//2 : len(layers)//2 + 2],
        "Final Layers": layers[-2:]
    }
    for label, group in stages.items():
        avg_norm = np.mean([stats[l]["mean_norm"] for l in group], axis=0)
        ax.plot(get_smooth_steps(steps), smooth_data(avg_norm), label=label, linewidth=2)
    
    ax.set_title("Key Norm Evolution (Smoothed)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # 2. Max Absolute Value (Outliers / Sink Detection)
    ax = axes[0, 1]
    for layer in [layers[0], layers[len(layers)//2], layers[-1]]:
        ax.plot(get_smooth_steps(steps), smooth_data(stats[layer]["max_abs"]), label=f"Layer {layer.split('_')[1]}", alpha=0.8)
    
    ax.set_title("Outlier Magnitude (Attention Sink Signal)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Max Abs Value")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # 3. Cosine Similarity (THE MESSY GRAPH - CLEANED)
    ax = axes[1, 0]
    # We group layers to see the systemic shift in latent density
    for label, group in stages.items():
        avg_sim = np.mean([stats[l]["cosine_sim"] for l in group], axis=0)
        ax.plot(get_smooth_steps(steps), smooth_data(avg_sim), label=label, linewidth=2)
        
    ax.set_title("Latent Redundancy (Avg Cosine Similarity)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Similarity Score")
    ax.set_ylim(-0.1, 0.6) # Standard scale for similarity
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # 4. Final Energy Distribution across all layers
    ax = axes[1, 1]
    final_norms = [stats[l]["mean_norm"][-1] for l in layers]
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    ax.bar(range(len(layers)), final_norms, color=colors, alpha=0.8)
    ax.set_title("Key Magnitude Profile (Last Step)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Final Norm")
    
    plt.tight_layout(pad=3.0)
    plt.savefig("research_results/key_analysis_report.png", dpi=120)
    print("\n📈 Enhanced clean report saved to research_results/key_analysis_report.png")

if __name__ == "__main__":
    run_experiment()
