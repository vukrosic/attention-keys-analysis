import json
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from research.track_keys import plot_results

def regenerate_plots():
    stats_path = "research_results/key_stats.json"
    if not os.path.exists(stats_path):
        print(f"❌ Error: {stats_path} not found.")
        return

    print("📊 Loading stats and regenerating graphs...")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    plot_results(stats)
    print("✅ Done!")

if __name__ == "__main__":
    regenerate_plots()
