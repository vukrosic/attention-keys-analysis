# H100 Run Plan — QK-Norm Causal Study

## What This Is

A controlled experiment to answer: **Does rank collapse in transformer attention come from QK-Norm's learned γ parameter, or from the RMS normalization itself?**

## How To Run

```bash
cd /root/llm-research-kit
python research/qk_norm_25m_study.py
```

That's it. One command. Everything is automated — training, probing, plotting, saving.

## What It Does

Trains a **1.5B parameter LLM** for **50M tokens** under 3 conditions:

| Condition | Tag | QK-Norm? | γ learnable? | What it isolates |
|:---|:---|:---|:---|:---|
| A | `QK` | Yes | Yes (learned) | Full QK-Norm effect |
| B | `QK_frozen` | Yes | No (frozen at 1.0) | RMS normalization only |
| C | `NoQK` | No | N/A | Baseline (Identity) |

Each condition: fresh model init (same seed), same data, same hyperparameters.

## How To Read The Results

The key plot is `2_key_pr_causal.png`:

- **If A ≠ B ≈ C** → γ (learned scaling) drives rank collapse
- **If A ≈ B ≠ C** → RMS normalization drives it, γ is innocent
- **If all three differ** → both contribute

Also check `4_causal_decomposition.png` which shows γ-effect vs norm-effect per layer.

## Config

| Parameter | Value |
|:---|:---|
| Model | 1.5B params, 32 layers, 16 Q heads, 8 KV heads, d_k=128 |
| Tokens | 50M per condition (150M total) |
| Batch size | 4 × 2048 tokens |
| Grad accumulation | 2 (effective 16,384 tokens/step) |
| Optimizer | Muon (2D weights) + AdamW (1D), constant LR |
| Probe freq | Every 120 forward steps (~50 data points per run) |
| Dataset | `/root/llm-research-kit/processed_data/pretrain_mix_26000000` |
| Seed | 42 |

## Time Estimate (H100)

- ~45 min per condition × 3 = **~2.5 hours total**
- Plotting adds ~2 min

## Output

All results go to `research_results/qk_norm_50m_study/`:

```
research_results/qk_norm_50m_study/
├── QK/results.json           # Condition A raw data
├── QK_frozen/results.json    # Condition B raw data
├── NoQK/results.json         # Condition C raw data
├── combined.json             # All 3 combined
├── 1_loss.png                # Train + val loss (solid/dashed)
├── 2_key_pr_causal.png       # THE MAIN RESULT
├── 3_per_layer.png           # Per-layer PR bars (3 conditions)
├── 4_causal_decomposition.png # Δ(A-B) vs Δ(B-C) per layer
└── 5_gamma_cv.png            # γ non-uniformity over training
```

## What Each JSON Contains

```json
{
  "tag": "QK",
  "tokens": [0, 983040, ...],       // tokens seen at each probe
  "loss": [null, 10.23, ...],        // training loss
  "val_loss": [null, 10.31, ...],    // validation loss (4 batch avg)
  "mean_pr": [85.2, 72.1, ...],     // mean Key PR across layers
  "layer_pr": {"0": [...], ...},     // per-layer Key PR
  "gamma_cv": {"0": [...], ...},     // per-layer γ coefficient of variation
  "gamma_final": {"0": [...], ...}   // full γ vectors at end (condition A only)
}
```

## Measurements

| Metric | What it is | Applied to |
|:---|:---|:---|
| **Key PR** | Participation ratio of post-norm key representations via SVD | All 3 conditions |
| **Train loss** | Cross-entropy on training batches | All 3 |
| **Val loss** | Cross-entropy on 4 val batches at each probe | All 3 |
| **γ-CV** | Coefficient of variation (std/mean) of RMSNorm γ | Conditions A, B |
| **γ final** | Full 128-dim γ vectors per layer | Condition A only |

## Key Files

| File | What |
|:---|:---|
| `research/qk_norm_25m_study.py` | Main script (training + plotting) |
| `research/svd_probe.py` | Probe that hooks k_norm, computes PR and γ stats |
| `models/layers.py` | Model architecture — lines 87-88 show where QK-Norm is applied |
| `configs/llm_config_1b.py` | Model config (1.5B params) |
| `training/trainer.py` | `setup_muon_optimizer()` function |

## Kill Criterion

If `|mean_PR(A) − mean_PR(B)| < 1.0` at the final checkpoint → γ is NOT the mechanism. Report as null result and pivot narrative to normalization.

## Known Limitations

- Single seed (no variance estimate)
- 50M tokens is early for 1.5B (0.3% of Chinchilla-optimal)
- K PR averaged over 8 KV heads, not Q heads (GQA architecture)
- Pre-RoPE hooking (approximation — RoPE is orthogonal per position)
- Dataset is 26M tokens, so we loop ~2× through it
