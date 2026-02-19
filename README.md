# QK-Norm and Rank Collapse in LLMs

This repository contains the code and results for studying **dimensional collapse** in transformer attention heads, specifically investigating how **QK-Normalization** (RMSNorm on query/key projections) affects the effective rank of key representations during pretraining.

## Key Findings

We trained a **1.5B parameter** model under three conditions for **~50M tokens** each:

| Condition | What it does | Final PR | Final Train Loss |
|:---|:---|:---:|:---:|
| **Learned γ** | Full QK-Norm (normalization + learnable γ) | **65.7** | 3.618 |
| **Frozen γ=1** | Normalization only (γ locked at 1.0) | 59.6 | **3.614** |
| **No QK-Norm** | No normalization at all | 63.3 | 3.683 |

**The main observations:**
1. **γ controls dimensional diversity (PR), normalization controls loss** — these are separable mechanisms
2. **PR does not predict model quality** — the model with the *lowest* PR achieves the *best* loss
3. **All models follow a U-shaped PR trajectory** — collapse → recovery → plateau
4. **Normalization without γ hurts rank** — Frozen γ=1 has worse PR than even no normalization

→ **Full write-up**: [`research/paper.md`](research/paper.md)

→ **Raw data & plots**: [`research_results/qk_norm_50m_study/`](research_results/qk_norm_50m_study/)

→ **500K pilot study**: [`research_results/qk_norm_500k_study/qk_norm_500k_report.md`](research_results/qk_norm_500k_study/qk_norm_500k_report.md)

## Repository Structure

```
├── research/
│   ├── qk_norm_25m_study.py    # Main 50M-token experiment (train + probe + plot)
│   ├── qk_norm_500k_study.py   # Earlier 500K-token pilot study
│   ├── split_panels.py         # Generates individual panel images from results JSON
│   └── svd_probe.py            # SVD-based rank measurement (Participation Ratio)
│
├── research/paper.md           # Full blog post with analysis and figures
│
├── research_results/
│   ├── qk_norm_50m_study/
│   │   ├── QK/results.json             # Learned γ condition (raw data + gamma_final)
│   │   ├── QK_frozen/results.json      # Frozen γ=1 condition (raw data)
│   │   ├── NoQK/results.json           # No QK-Norm condition (raw data)
│   │   ├── 1_loss.png                  # Training & validation loss plot
│   │   └── 2_key_pr_causal.png         # PR trajectory plot (the main figure)
│   └── qk_norm_500k_study/            # Earlier pilot study results
│
├── models/
│   ├── llm.py           # MinimalLLM model (1.5B params, Gemma-style)
│   ├── layers.py        # TransformerBlock with GQA attention + QK-Norm
│   └── components.py    # Feedforward components
│
├── configs/
│   ├── llm_config.py      # Base LLMConfig dataclass
│   ├── llm_config_1b.py   # 1B-scale config (2048 d_model, 32 layers, 16 heads)
│   └── dataset_config.py  # Data loading configuration
│
├── data/
│   └── loader.py        # Dataset loading + tokenization
│
├── training/
│   ├── trainer.py       # Training loop + Muon optimizer setup
│   └── evaluation.py    # Model evaluation utilities
│
├── optimizers/
│   └── muon.py          # Muon optimizer implementation
│
├── utils/
│   ├── helpers.py       # Seed setting, time formatting
│   └── logger.py        # Logging setup
│
├── train_llm.py         # General training entrypoint (provides prepare_datasets)
└── requirements.txt     # Python dependencies
```

## Reproducing the Experiment

### Prerequisites

**Hardware:** Any NVIDIA GPU with ≥24GB VRAM. We ran on an H100 80GB, but the script is configurable for smaller GPUs. Here's how to adjust for your hardware:

| GPU | VRAM | Recommended Settings |
|:---|:---:|:---|
| **H100 / A100 80GB** | 80GB | Default settings (`BATCH_SIZE=8`, `GRAD_ACCUM=4`, `compile_model=True`) |
| **A100 40GB** | 40GB | Reduce `BATCH_SIZE=4`, increase `GRAD_ACCUM=8` to keep the same effective batch |
| **RTX 4090 / 3090** | 24GB | `BATCH_SIZE=1`, `GRAD_ACCUM=32`, set `compile_model=False` if OOM persists |
| **A10G / L4** | 24GB | Same as 4090, consider `gradient_checkpointing=True` (already on by default) |
| **Multi-GPU** | — | Not supported out of the box — the script runs single-GPU. Wrap in DDP manually if needed |

These parameters are at the top of `research/qk_norm_25m_study.py`:

```python
TARGET_TOKENS = 50_000_000    # Tokens per condition — reduce for faster testing
BATCH_SIZE = 8                # ← Lower this for smaller GPUs
GRAD_ACCUM = 4                # ← Increase proportionally to keep effective batch stable
```

The model config (`compile_model`, `gradient_checkpointing`) is set inside the `run()` function. To disable `torch.compile` (e.g., on older GPUs), change `config.compile_model = False`.

**Effective batch size** = `BATCH_SIZE × GRAD_ACCUM × 2048` (sequence length). The default is 65,536 tokens/step. Keeping this constant across hardware ensures comparable results.

**Software:**

```bash
# Clone the repo
git clone https://github.com/vukrosic/attention-keys-analysis.git
cd attention-keys-analysis

# Install dependencies (requires Python 3.10+ and PyTorch 2.0+)
pip install -r requirements.txt
```

The `requirements.txt` includes: `datasets`, `transformers`, `torchtune`, `torchao`, `matplotlib`.

You also need PyTorch with CUDA support. Install it separately if needed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 1: Prepare the Data

The experiment uses a 1B-token subset of FineWeb/Cosmopedia, preprocessed and tokenized. The data must be at:
```
processed_data/pretrain_1B/
```

This is a HuggingFace `datasets` directory created by tokenizing the raw data with the Gemma tokenizer (vocab size 49,152) at sequence length 2048. If you need to recreate it, use `train_llm.py` which calls `prepare_datasets()` from `data/loader.py` with the appropriate `DataConfig`.

### Step 2: Run the Experiment

```bash
python research/qk_norm_25m_study.py
```

This single script does everything:

1. **Trains 3 models sequentially** (each ~50M tokens):
   - Condition A: QK-Norm with learned γ
   - Condition B: QK-Norm with frozen γ=1
   - Condition C: No QK-Norm
2. **Probes every 120 steps** (~2M tokens) — computes SVD on key representations and records Participation Ratio per layer
3. **Saves results** to `research_results/qk_norm_50m_study/{QK,QK_frozen,NoQK}/results.json`
4. **Generates plots** (loss curves, PR trajectories, per-layer bars, γ-CV evolution)

See the **Prerequisites** section above for adjusting `BATCH_SIZE`, `GRAD_ACCUM`, and `TARGET_TOKENS` for your GPU.

**Runtime:** ~4–6 hours total on a single H100 (3 runs × ~1.5–2 hours each).

### Step 3: View Results

After the run completes, results are saved to:

```
research_results/qk_norm_50m_study/
├── QK/results.json             # Learned γ: loss, PR, per-layer PR, gamma_cv, gamma_final
├── QK_frozen/results.json      # Frozen γ=1: loss, PR, per-layer PR
├── NoQK/results.json           # No QK-Norm: loss, PR, per-layer PR
├── 1_loss.png                  # Training & validation loss comparison
├── 2_key_pr_causal.png         # PR trajectory (the main figure)
├── 3_per_layer.png             # Per-layer PR at final checkpoint
├── 4_causal_decomposition.png  # γ effect vs. normalization effect per layer
└── 5_gamma_cv.png              # γ coefficient of variation over training
```

Each `results.json` contains:
- `tokens`: list of token counts at each probe point
- `loss` / `val_loss`: training and validation loss at each probe
- `mean_pr`: mean Participation Ratio across all layers
- `layer_pr`: per-layer PR over time (dict of layer_idx → list of PR values)
- `gamma_cv`: per-layer γ coefficient of variation over time (QK and QK_frozen only)
- `gamma_final`: final γ values for all 32 layers × 128 dimensions (QK only)

### Reproducing the 500K Pilot Study

The earlier pilot study (2 conditions × 500K tokens) can be reproduced with:

```bash
python research/qk_norm_500k_study.py
```

This is faster (~30 min on A100) but only compares QK-Norm vs. No QK-Norm (no frozen γ ablation).

### Re-generating Plots

To regenerate individual panel images from existing results without retraining:

```bash
python research/split_panels.py
```

## Method

We measure the **Participation Ratio (PR)** of post-RoPE key representations at each attention layer:

$$PR = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}$$

where $\sigma_i$ are singular values of the key matrix $K \in \mathbb{R}^{n \times d_k}$ ($n$ = tokens in eval batch, $d_k$ = 128).

- **PR = 128** → full rank (all dimensions carry equal variance)
- **PR = 1** → total collapse (one dimension dominates)
- **PR ≈ 60** → roughly 60 of 128 dimensions carry significant variance (what we observe)

**Important:** PR measures geometric spread of variance, not information content. A low-variance dimension could still encode critical features. Our experiment shows that lower PR does not correspond to worse model loss — this dissociation is a central finding.

## License

MIT
