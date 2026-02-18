
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig

def get_spectrum(model, layer_idx):
    attn = model.transformer_blocks[layer_idx].attention
    q_size = attn.q_size
    kv_size = attn.kv_size
    d_k = attn.d_k
    n_heads = attn.n_heads
    
    # Extract K weights: [kv_size, d_model]
    W_K = attn.qkvo_proj[q_size:q_size+kv_size, :].detach().cpu().float()
    
    # We want head-wise spectrum for better resolution
    # Reshape kv weights to [n_heads, d_k, d_model]
    W_K_heads = W_K.reshape(n_heads, d_k, -1)
    
    all_s = []
    for h in range(n_heads):
        # Singular values of [64, 512] matrix
        s = torch.linalg.svdvals(W_K_heads[h])
        all_s.append(s.numpy())
        
    # Return average spectrum across heads
    return np.mean(all_s, axis=0)

def plot_spectrum_comparison():
    config = LLMConfig()
    device = "cpu"
    
    # Load Models
    config_qk = LLMConfig(use_qk_norm=True)
    model_qk = MinimalLLM(config_qk)
    path_qk = "research_results/muon_20m_study/muon_qk_20m/model_final.pt"
    model_qk.load_state_dict(torch.load(path_qk, map_location=device, weights_only=True))
    
    config_noqk = LLMConfig(use_qk_norm=False)
    model_noqk = MinimalLLM(config_noqk)
    path_noqk = "research_results/muon_20m_study/muon_no_qk_20m/model_final.pt"
    model_noqk.load_state_dict(torch.load(path_noqk, map_location=device, weights_only=True))
    
    # Layers to check
    layers_to_plot = [5, 11, 20] # Early, Mid, Late
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, l_idx in enumerate(layers_to_plot):
        s_qk = get_spectrum(model_qk, l_idx)
        s_noqk = get_spectrum(model_noqk, l_idx)
        
        # Normalize by max singular value for better distribution comparison
        s_qk_norm = s_qk / s_qk[0]
        s_noqk_norm = s_noqk / s_noqk[0]
        
        axes[i].plot(s_qk_norm, 'r-', label="With QK-Norm", linewidth=2)
        axes[i].plot(s_noqk_norm, 'b-', label="No QK-Norm", linewidth=2)
        
        axes[i].set_title(f"Layer {l_idx} Spectrum")
        axes[i].set_xlabel("Singular Value Index")
        axes[i].set_ylabel("Normalized Magnitude")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
    plt.suptitle("Spectral Decay Comparison: QK-Norm vs No QK-Norm (25M Tokens)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = "research_results/muon_20m_study/spectral_dive.png"
    plt.savefig(out_path, dpi=200)
    print(f"✅ Spectral plot saved to {out_path}")

if __name__ == "__main__":
    plot_spectrum_comparison()
