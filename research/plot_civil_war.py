import json
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(data, window=5):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_civil_war(stats_path):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    layers = sorted(stats.keys(), key=lambda x: int(x.split('_')[1]))
    steps = stats[layers[0]]["steps"]
    num_steps = len(steps)
    
    os.makedirs("research_results/plots", exist_ok=True)
    
    # --- 1. Head Norm Divergence (Layer 12 as example) ---
    plt.figure(figsize=(10, 6))
    target_layer = layers[len(layers)//2]
    layer_data = stats[target_layer]["head_stats"]
    for h, h_stats in enumerate(layer_data):
        plt.plot(steps, h_stats["l2_norm"], label=f"Head {h}", alpha=0.7)
    plt.title(f"Inter-head Norm Divergence ({target_layer})", fontsize=14, fontweight='bold')
    plt.xlabel("Steps")
    plt.ylabel("L2 Norm")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("research_results/plots/norm_divergence.png")
    plt.close()

    # --- 2. Entropy vs Norm (Phase 2 Scatter) ---
    plt.figure(figsize=(10, 8))
    # Pick a few representative steps: early, middle, late
    sample_indices = [0, num_steps//4, num_steps//2, -1]
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    labels = ["Start", "Early", "Mid", "Late"]
    
    for idx, color, label in zip(sample_indices, colors, labels):
        norms = []
        entropies = []
        for l in layers:
            for h_stats in stats[l]["head_stats"]:
                norms.append(h_stats["l2_norm"][idx])
                entropies.append(h_stats["entropy"][idx])
        plt.scatter(norms, entropies, c=color, label=label, alpha=0.5, edgecolor='none')
    
    plt.title("The 'Civil War' Scatter: Entropy vs Key Norm", fontsize=14, fontweight='bold')
    plt.xlabel("Key Norm (L2)")
    plt.ylabel("Attention Entropy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("research_results/plots/entropy_norm_scatter.png")
    plt.close()

    # --- 3. Effective Rank (Participation Ratio) Evolution ---
    plt.figure(figsize=(10, 6))
    for l in [layers[1], layers[len(layers)//2], layers[-2]]:
        avg_pr = np.mean([h["pr"] for h in stats[l]["head_stats"]], axis=0)
        plt.plot(steps, avg_pr, label=f"Layer {l.split('_')[1]}")
    plt.title("Dimensional Focus: Participation Ratio (Effective Rank)", fontsize=14, fontweight='bold')
    plt.xlabel("Steps")
    plt.ylabel("PR Score (Lower = More Focused)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("research_results/plots/effective_rank.png")
    plt.close()

    # --- 4. Gain Vector Gamma Evolution ---
    l_idx = len(layers)//2
    gamma_history = np.array(stats[layers[l_idx]]["gamma"]) # [Steps, D_k]
    plt.figure(figsize=(12, 6))
    plt.imshow(gamma_history.T, aspect='auto', cmap='magma', interpolation='nearest')
    plt.colorbar(label='Gain Value')
    plt.title(f"Weight Evolution of k_norm.gamma ({layers[l_idx]})", fontsize=14, fontweight='bold')
    plt.xlabel("Steps")
    plt.ylabel("Dimension Index")
    plt.savefig("research_results/plots/gamma_heatmap.png")
    plt.close()

    print(f"📊 Visualizations saved to research_results/plots/")

if __name__ == "__main__":
    plot_civil_war("research_results/civil_war_stats.json")
