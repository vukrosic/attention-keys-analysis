import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.getcwd())
from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets
from torch.utils.data import DataLoader

def run_rank_surgery():
    print("🧠 Starting Rank Collapse Surgery Experiment")
    
    # 1. Setup - we'll train briefly to get a specialized model
    config = LLMConfig()
    config.train_tokens = 2_000_000 # Enough to see collapse
    config.compile_model = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinimalLLM(config).to(device)
    
    # Load data for evaluation
    data_cfg = DataConfig(dataset_path="auto", num_samples=5000)
    tokenizer = setup_tokenizer(data_cfg)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=8)
    
    # Train for a bit to ensure we have something to prune
    from training.trainer import setup_muon_optimizer
    optimizers = setup_muon_optimizer(model, config)
    
    print("Training briefly to establish dimensional collapse...")
    model.train()
    tokens = 0
    pbar = tqdm(total=config.train_tokens)
    while tokens < config.train_tokens:
        for batch in DataLoader(train_ds, batch_size=4):
            if tokens >= config.train_tokens: break
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1), ignore_index=-100)
            loss.backward(); [opt.step() for opt in optimizers]; [opt.zero_grad() for opt in optimizers]
            tokens += x.numel(); pbar.update(x.numel())
    pbar.close()

    # 2. THE SURGERY
    model.eval()
    
    def get_val_loss():
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i > 20: break # Quick eval
                x, y = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(x)
                total_loss += F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1), ignore_index=-100).item()
        return total_loss / 21

    baseline_loss = get_val_loss()
    print(f"\n📊 Baseline Val Loss: {baseline_loss:.4f}")

    # Percentages to prune (keeping the TOP dimensions)
    prune_percents = np.linspace(0, 95, 20)
    losses = []
    
    # We will prune dimensions globally across all layers for simplicity
    # Or per-layer based on their own gamma weights
    
    print("Performing surgery (Zeroing low-gain dimensions)...")
    original_weights = [block.attention.k_norm.weight.data.clone() for block in model.transformer_blocks]
    
    for p in tqdm(prune_percents):
        # Apply pruning
        for i, block in enumerate(model.transformer_blocks):
            w = original_weights[i].clone()
            # Sort by absolute magnitude
            mags = torch.abs(w)
            # Find threshold for bottom p percent
            threshold = torch.quantile(mags, p/100.0)
            mask = (mags >= threshold).float()
            block.attention.k_norm.weight.data = w * mask
            
        current_loss = get_val_loss()
        losses.append(current_loss)
        
    # Reset weights
    for i, block in enumerate(model.transformer_blocks):
        block.attention.k_norm.weight.data = original_weights[i]

    # 3. Correlation Experiment (Inter-layer Coordinate System)
    gamma_matrix = torch.stack([block.attention.k_norm.weight.data for block in model.transformer_blocks])
    # Compute similarity between layers
    sim_matrix = F.cosine_similarity(gamma_matrix.unsqueeze(1), gamma_matrix.unsqueeze(0), dim=-1).cpu().numpy()

    # PILOT PLOTTING
    os.makedirs("research_results/plots_extra", exist_ok=True)
    
    # Plot Surgery
    plt.figure(figsize=(10, 6))
    plt.plot(prune_percents, losses, 'o-', linewidth=2, color='#e74c3c')
    plt.axhline(y=baseline_loss, color='gray', linestyle='--', label='Baseline')
    plt.title("The 'Surgery' Test: Impact of Dimensional Pruning", fontsize=14, fontweight='bold')
    plt.xlabel("Percentage of Dimensions Zeroed (Bottom Gain)")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("research_results/plots_extra/surgery_impact.png")
    
    # Plot Correlation
    plt.figure(figsize=(8, 7))
    plt.imshow(sim_matrix, cmap='magma')
    plt.colorbar(label='Cosine Similarity')
    plt.title("Inter-Layer Coordinate Consistency", fontsize=14, fontweight='bold')
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.savefig("research_results/plots_extra/layer_correlation.png")

    print(f"\n✅ Supplemental experiments complete.")
    print(f"📈 Result images: research_results/plots_extra/")

if __name__ == "__main__":
    run_rank_surgery()
