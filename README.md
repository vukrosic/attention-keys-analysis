# QK-Norm and Rank Collapse in LLMs

This repository contains the code and results for studying **dimensional collapse** in transformer attention heads, specifically investigating how **QK-Normalization** (RMSNorm on query/key projections) affects the effective rank of key representations during pretraining.

## Key Finding

At 500K tokens of training on a 1.5B parameter model, **QK-Norm shows higher Participation Ratio** (spectral uniformity) compared to an identical model without it. The effect is small (+0.75 PR out of 128, single seed) but consistent across all 32 layers, with the strongest signal in deeper layers (+1.58 at Layer 30). Rank collapse begins within the first 100K tokens and follows a strong depth gradient (~122 at Layer 0 to ~106 at Layer 30).

→ **Full report**: [`research_results/qk_norm_500k_study/qk_norm_500k_report.md`](research_results/qk_norm_500k_study/qk_norm_500k_report.md)

## Repository Structure

```
├── research/
│   ├── qk_norm_500k_study.py    # Main experiment script (train + probe + plot + report)
│   ├── split_panels.py          # Generates individual panel images from results JSON
│   └── svd_probe.py             # SVD-based rank measurement (Participation Ratio)
│
├── research_results/
│   └── qk_norm_500k_study/
│       ├── qk_norm_500k_report.md      # Full research report with analysis
│       ├── combined_results.json       # Raw numerical data (PR, loss, per-layer)
│       ├── panel_a_loss.png            # Training loss comparison
│       ├── panel_b_pr_trajectory.png   # Mean PR over time
│       ├── panel_c_per_layer.png       # Per-layer PR at final checkpoint
│       ├── panel_d_delta.png           # QK-Norm effect per layer
│       └── layer_pr_heatmap_500k.png   # Layer × time PR heatmap
│
├── models/
│   ├── llm.py           # MinimalLLM model (1.5B params)
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

```bash
pip install -r requirements.txt
```

You also need the preprocessed training data at:
```
processed_data/pretrain_mix_26000000/
```

### Run the Experiment

```bash
# Full experiment: train with and without QK-Norm, generate plots + report
python research/qk_norm_500k_study.py

# Re-generate individual panel images from existing results
python research/split_panels.py
```

The experiment trains a **1.5B parameter** model twice (once with QK-Norm, once without) for **500K tokens** each, measuring the Participation Ratio via SVD every 50 steps.

**Runtime**: ~30 minutes on a single A100 GPU.

## Method

We measure the **Participation Ratio (PR)** of post-RoPE key representations:

$$PR = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}$$

where $\sigma_i$ are singular values of the key matrix $K \in \mathbb{R}^{n \times d_k}$.

- **PR = 128** → full rank (all dimensions used equally)
- **PR = 1** → total collapse (one dimension dominates)

## License

MIT
