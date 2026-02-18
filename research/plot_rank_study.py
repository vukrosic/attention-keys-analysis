import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_rank_comparison(study_dir="research_results/rank_study"):
    study_path = Path(study_dir)
    if not study_path.exists():
        print(f"Directory {study_dir} not found.")
        return

    # Find all experiment folders
    exp_folders = [f for f in study_path.iterdir() if f.is_dir()]
    
    # 1. Trajectory Plot: PR average over tokens
    plt.figure(figsize=(12, 7))
    
    for exp in exp_folders:
        probe_file = exp / "probe_results.json"
        if not probe_file.exists(): continue
        
        with open(probe_file, "r") as f:
            data = json.load(f)
            
        # Extract steps and avg PR across all layers
        steps = sorted([int(k) for k in data.keys()])
        pr_avgs = []
        for s in steps:
            layer_data = data[str(s)]
            avg_pr = np.mean([ld["pr_avg"] for ld in layer_data.values()])
            pr_avgs.append(avg_pr)
            
        plt.plot(steps, pr_avgs, 'o-', label=exp.name)
        
    plt.title("Dimensional Collapse Trajectory (Participation Ratio)", fontsize=14)
    plt.xlabel("Tokens Seen")
    plt.ylabel("Avg Participation Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(study_path / "pr_trajectory.png")
    
    # 2. Per-Layer Heatmap for final checkpoint
    for exp in exp_folders:
        probe_file = exp / "probe_results.json"
        if not probe_file.exists(): continue
        
        with open(probe_file, "r") as f:
            data = json.load(f)
            
        final_step = str(max([int(k) for k in data.keys()]))
        layer_data = data[final_step]
        n_layers = len(layer_data)
        
        layers = sorted([int(k) for k in layer_data.keys()])
        pr_values = [layer_data[str(l)]["pr_avg"] for l in layers]
        
        plt.figure(figsize=(10, 4))
        plt.bar(layers, pr_values, color='skyblue')
        plt.title(f"Final Participation Ratio per Layer: {exp.name}")
        plt.xlabel("Layer Index")
        plt.ylabel("PR (Max 64)")
        plt.axhline(y=64, color='r', linestyle='--', label='Full Rank (64)')
        plt.legend()
        plt.savefig(study_path / f"layer_rank_{exp.name}.png")

    print(f"✅ Plots generated in {study_dir}")

if __name__ == "__main__":
    plot_rank_comparison()
