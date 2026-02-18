
import json
import matplotlib.pyplot as plt
import os

def plot_repro_comparison():
    qk_path = "research_results/repro_1b/repro_1b_QK/results.json"
    noqk_path = "research_results/repro_1b/repro_1b_NoQK/results.json"
    
    if not os.path.exists(qk_path) or not os.path.exists(noqk_path):
        print("Waiting for results...")
        return

    with open(qk_path, "r") as f: qk = json.load(f)
    with open(noqk_path, "r") as f: no_qk = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Val Loss
    ax1.plot(qk["tokens"], qk["val_loss"], label="With QK-Norm", color='red')
    ax1.plot(no_qk["tokens"], no_qk["val_loss"], label="Without QK-Norm", color='blue')
    ax1.set_title("Validation Loss (1.5B Model)")
    ax1.set_xlabel("Tokens")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # Mean PR
    ax2.plot(qk["tokens"], qk["mean_pr"], label="With QK-Norm", color='red', marker='o')
    ax2.plot(no_qk["tokens"], no_qk["mean_pr"], label="Without QK-Norm", color='blue', marker='s')
    ax2.set_title("Mean Participation Ratio (1.5B Model)")
    ax2.set_xlabel("Tokens")
    ax2.set_ylabel("PR (Max 128)")
    ax2.set_ylim(0, 130)
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs("research_results/repro_1b/plots", exist_ok=True)
    plt.savefig("research_results/repro_1b/plots/comparison_1b.png")
    plt.show()
    print("📈 Comparison Plot saved: research_results/repro_1b/plots/comparison_1b.png")

if __name__ == "__main__":
    plot_repro_comparison()
