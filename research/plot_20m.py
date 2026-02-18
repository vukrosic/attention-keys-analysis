
import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_20m_results(base_dir="research_results/muon_20m_study"):
    base_path = Path(base_dir)
    experiments = ["muon_qk_20m", "muon_no_qk_20m"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for exp in experiments:
        metrics_file = base_path / exp / "metrics.json"
        if not metrics_file.exists():
            print(f"Skipping {exp}, file not found.")
            continue
            
        with open(metrics_file, "r") as f:
            data = json.load(f)
            
        tokens = [d["tokens"] / 1e6 for d in data]
        val_loss = [d["val_loss"] for d in data]
        pr = [d["pr"] for d in data]
        
        label = "QK-Norm (LR 0.012)" if "qk" in exp and "no_qk" not in exp else "No QK-Norm (LR 0.012)"
        color = "blue" if "no_qk" in exp else "red"
        
        ax1.plot(tokens, val_loss, label=label, color=color, linewidth=2)
        ax2.plot(tokens, pr, label=label, color=color, linewidth=2)
        
    ax1.set_title("Validation Loss (20M Tokens)")
    ax1.set_xlabel("Tokens (Millions)")
    ax1.set_ylabel("Cross Entropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Participation Ratio (Rank)")
    ax2.set_xlabel("Tokens (Millions)")
    ax2.set_ylabel("PR (Max 64)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("research_results/muon_20m_study/comparison_20m.png", dpi=200)
    print("✅ Plot saved to research_results/muon_20m_study/comparison_20m.png")

if __name__ == "__main__":
    plot_20m_results()
