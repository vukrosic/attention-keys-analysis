import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("Starting plot generation...")

experiments = {
    'A1: Muon + QK-Norm': ('research_results/rank_study/A1_Baseline_QK_Muon/probe_results.json', '#ff6b6b', '--', 'o'),
    'A2: Muon (no QK-Norm)': ('research_results/rank_study/A2_NoQK_Muon/probe_results.json', '#51cf66', '-', 's'),
    'B1: AdamW + QK-Norm': ('research_results/rank_study/B1_Baseline_QK_AdamW/probe_results.json', '#ff922b', '--', '^'),
    'B2: AdamW (no QK-Norm)': ('research_results/rank_study/B2_NoQK_AdamW/probe_results.json', '#339af0', '-', 'D'),
}

os.chdir('/root/llm-research-kit')

# ========== PLOT 1: Trajectory ==========
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

for name, (path, color, ls, marker) in experiments.items():
    with open(path) as f:
        data = json.load(f)
    
    milestones = sorted(data.keys(), key=lambda x: int(x))
    tokens = [int(m) / 1_000_000 for m in milestones]
    avg_prs = []
    for m in milestones:
        layers = data[m]
        prs = [layers[str(i)]['pr_avg'] for i in range(22)]
        avg_prs.append(sum(prs) / len(prs))
    
    ax.plot(tokens, avg_prs, color=color, linestyle=ls, marker=marker, 
            markersize=8, linewidth=2.5, label=name, zorder=5)

ax.annotate('Recovery\nphase', xy=(2, 49), xytext=(5, 56),
            color='#51cf66', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#51cf66', lw=1.5),
            ha='center')

ax.annotate('Collapse\nacceleration', xy=(8, 26), xytext=(12, 10),
            color='#ff922b', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ff922b', lw=1.5),
            ha='center')

ax.set_xlabel('Tokens (Millions)', fontsize=14, color='white', fontweight='bold')
ax.set_ylabel('Average Participation Ratio (Effective Rank)', fontsize=14, color='white', fontweight='bold')
ax.set_title('Dimensional Collapse: Optimizer × QK-Normalization\n22-Layer Transformer, 25M Tokens', 
             fontsize=16, color='white', fontweight='bold', pad=15)

ax.tick_params(colors='#8b949e', labelsize=11)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.15, color='#8b949e')

ax.axhline(y=64, color='#8b949e', linestyle=':', alpha=0.4, linewidth=1)
ax.text(0.3, 64.5, 'Max rank (d_k=64)', color='#8b949e', fontsize=9, alpha=0.6)

ax.set_ylim(5, 68)
ax.set_xlim(-0.5, 26)

legend = ax.legend(fontsize=12, loc='center right', 
                   facecolor='#21262d', edgecolor='#30363d',
                   labelcolor='white', framealpha=0.95)

plt.tight_layout()
plt.savefig('research_results/rank_study/pr_trajectory_4way.png', dpi=200, 
            facecolor='#0d1117', bbox_inches='tight')
print('Saved pr_trajectory_4way.png')
plt.close()

# ========== PLOT 2: Layer-wise comparison ==========
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor('#0d1117')
fig2.suptitle('Layer-Wise Effective Rank at 25M Tokens', fontsize=16, color='white', fontweight='bold')

configs = [
    ('A2: Muon (no QK-Norm)', 'research_results/rank_study/A2_NoQK_Muon/probe_results.json', '#51cf66', 'Avg PR=51.2'),
    ('A1: Muon + QK-Norm', 'research_results/rank_study/A1_Baseline_QK_Muon/probe_results.json', '#ff6b6b', 'Avg PR=30.0'),
    ('B2: AdamW (no QK-Norm)', 'research_results/rank_study/B2_NoQK_AdamW/probe_results.json', '#339af0', 'Avg PR=32.8'),
    ('B1: AdamW + QK-Norm', 'research_results/rank_study/B1_Baseline_QK_AdamW/probe_results.json', '#ff922b', 'Avg PR=16.2'),
]

for idx, (name, path, color, subtitle) in enumerate(configs):
    ax = axes[idx // 2][idx % 2]
    ax.set_facecolor('#161b22')
    
    with open(path) as f:
        data = json.load(f)
    final_key = max(data.keys(), key=lambda x: int(x))
    layers = data[final_key]
    prs = [layers[str(i)]['pr_avg'] for i in range(22)]
    
    bars = ax.bar(range(22), prs, color=color, alpha=0.85, edgecolor=color, linewidth=0.5)
    
    for i, pr in enumerate(prs):
        if pr < 10:
            bars[i].set_color('#e74c3c')
            bars[i].set_edgecolor('#c0392b')
    
    ax.set_title(f'{name}\n({subtitle})', fontsize=12, color='white', fontweight='bold')
    ax.set_xlabel('Layer', fontsize=10, color='#8b949e')
    ax.set_ylabel('PR', fontsize=10, color='#8b949e')
    ax.set_ylim(0, 65)
    ax.axhline(y=10, color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1)
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, axis='y', color='#8b949e')

plt.tight_layout()
plt.savefig('research_results/rank_study/layer_comparison_4way.png', dpi=200,
            facecolor='#0d1117', bbox_inches='tight')
print('Saved layer_comparison_4way.png')
plt.close()

print("Done!")
