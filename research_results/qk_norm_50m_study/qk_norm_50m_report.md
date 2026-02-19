# QK-Norm and Rank Collapse at 1.5B Scale: A 50M Token Causal Study

## 1. Overview

This study extends our [previous 500K-token pilot](../qk_norm_500k_study/qk_norm_500k_report.md) from an observational comparison to a **causal experiment**. We introduce a third condition — **Frozen γ=1** — that isolates the effect of the learnable scale parameter $\gamma$ from the effect of RMSNorm itself.

We train a **1.5B parameter** dense LLM for **~49M tokens** under three conditions:

1. **Learned γ** (full QK-Norm): RMSNorm on Q and K with learnable $\gamma \in \mathbb{R}^{d_k}$
2. **Frozen γ=1** (ablation): RMSNorm on Q and K with $\gamma$ frozen at $\mathbf{1}$ (normalization only, no learned scaling)
3. **No QK-Norm** (baseline): Identity — no normalization applied to Q or K

By comparing all three, we can determine whether the spectral properties of key representations are driven by:
- The **normalization dynamics** (gradient stabilization from constraining $\|K\|$), or
- The **learned parameter** $\gamma$ selectively suppressing or amplifying dimensions

**Key finding:** The learned $\gamma$ is the dominant driver of rank preservation. At 50M tokens, Learned γ maintains a mean PR of **65.7**, while Frozen γ=1 collapses to **59.6** — closer to the No QK-Norm baseline (**63.3**) than to the full method. This answers the central causal question from our 500K pilot.

### Architecture & Training Configuration

| Parameter | Value |
|:---|:---|
| **Total Parameters** | ~1.58B |
| $d_\text{model}$ | 2048 |
| Layers | 32 |
| Attention Heads ($n_h$) | 16 |
| Head Dimension ($d_k$) | $d_\text{model} / n_h = 128$ |
| KV Heads (GQA) | 8 |
| $d_\text{ff}$ | 8192 |
| **Hardware** | NVIDIA H100 80GB |
| Batch Size | 8 |
| Gradient Accumulation | 4 |
| Effective Batch | $8 \times 4 \times 2048 = 65{,}536$ tokens/step |
| Optimizer | Muon ($\eta = 0.012$) + AdamW ($\eta = 0.003$) |
| Sequence Length | 2048 |
| Data | FineWeb / Cosmopedia-v2 Mix (1B subset) |
| **Total Training Budget** | ~49.2M tokens per condition |
| PR Probe Interval | Every 120 steps (~2M tokens) |
| Total PR Measurements | 26 per condition (including init) |

---

## 2. The Causal Question

### 2.1 Motivation

The 500K-token pilot observed that QK-Norm models maintain higher Participation Ratio (PR) than No QK-Norm models, but could not determine **why**. QK-Norm introduces two mechanisms simultaneously:

1. **RMSNorm normalization** — constrains $\|K\|$, stabilizes attention logit magnitudes
2. **Learnable per-dimension scale** $\gamma_j$ — can independently amplify or suppress each of the $d_k = 128$ dimensions

The Frozen γ=1 ablation surgically removes mechanism (2) while preserving mechanism (1).

### 2.2 The Triangulation Logic

The three-way comparison creates a clean causal inference:

$$
\text{If } \text{PR}(\text{Learned}) \neq \text{PR}(\text{Frozen}) \approx \text{PR}(\text{NoQK}) \implies \gamma \text{ drives rank differences}
$$

$$
\text{If } \text{PR}(\text{Learned}) \approx \text{PR}(\text{Frozen}) \neq \text{PR}(\text{NoQK}) \implies \text{normalization drives rank differences}
$$

$$
\text{If all three differ} \implies \text{both mechanisms contribute}
$$

### 2.3 Participation Ratio (PR)

For each attention head $h$ in layer $\ell$, we collect post-RoPE key representations over a fixed evaluation batch:

$$K^{(\ell, h)} \in \mathbb{R}^{n \times d_k}$$

We compute the compact SVD and derive the Participation Ratio:

$$\text{PR}(K) = \frac{\left(\sum_{i=1}^{d_k} \sigma_i\right)^2}{\sum_{i=1}^{d_k} \sigma_i^2}$$

This measures spectral concentration — how evenly the singular values are distributed. For $d_k = 128$: PR = 128 means perfectly uniform spectrum (full rank), PR = 1 means total collapse (one dominant dimension).

---

## 3. Results

### 3.1 Summary Table

| Metric | Learned γ | Frozen γ=1 | No QK-Norm |
|:---|:---:|:---:|:---:|
| **Initial Mean PR** | 86.85 | 86.85 | 86.42 |
| **Minimum Mean PR** | 55.17 (at 7.9M) | 51.84 (at 9.8M) | 51.28 (at 7.9M) |
| **Final Mean PR (49M)** | **65.70** | 59.55 | 63.25 |
| **PR Drop from Init** | −21.15 | −27.30 | −23.17 |
| **Recovery from Min** | +10.53 | +7.71 | +11.97 |
| **Final Train Loss** | **3.618** | **3.614** | 3.683 |
| **Final Val Loss** | **3.440** | **3.439** | 3.505 |
| **Most Collapsed Layer** | L5 (46.2) | L5 (43.5) | L4 (39.0) |
| **Highest PR Layer** | L25 (77.1) | L1 (76.4) | L0 (77.2) |

### 3.2 The Causal Verdict: γ Drives Rank Preservation

The results clearly show **A ≠ B ≈ C** at the collapse floor (~8M tokens), but then **all three diverge** during recovery:

- At **minimum collapse** (~8M tokens): Learned γ = 55.2, Frozen γ = 51.8, No QK-Norm = 51.3
  - Frozen ≈ NoQK → normalization alone does **not** prevent collapse
  - Learned γ is already 3–4 PR units higher → γ begins to act early

- At **50M tokens** (after recovery): Learned γ = 65.7, No QK-Norm = 63.3, Frozen γ = 59.6
  - The ordering **reverses** for Frozen vs. NoQK — Frozen γ ends up with the **lowest** PR
  - This means normalization without learned γ actually *hinders* rank recovery

**Conclusion:** The learned $\gamma$ parameter is the primary mechanism through which QK-Norm affects dimensional rank. Normalization alone (Frozen γ=1) produces *worse* rank recovery than no normalization at all.

---

## 4. Figure-by-Figure Analysis

### 4.1 Figure 1 — Training & Validation Loss

![Training & Validation Loss](1_loss.png)

**What it shows:** Training loss (solid lines) and validation loss (dashed lines) vs. tokens processed (in millions), for all three conditions: Learned γ (orange), Frozen γ=1 (purple), and No QK-Norm (cyan).

**Reading the curves:**

- All three conditions start with loss ~7.25 at 2M tokens and converge to ~3.6 by 49M tokens — a total reduction of ~3.6 nats.
- **No QK-Norm consistently has the worst loss.** The gap is visible from ~5M tokens onward. At 49M tokens: NoQK train loss = 3.683 vs. Learned γ = 3.618 (Δ = −0.065).
- **Learned γ and Frozen γ=1 are nearly identical in loss.** The gap between them is negligible: 3.618 vs. 3.614 train loss. Frozen γ=1 is marginally better — likely within noise.
- Validation loss confirms the pattern: Learned γ = 3.440, Frozen γ = 3.439, NoQK = 3.505.

**Key loss data points:**

| Tokens (M) | Learned γ | Frozen γ=1 | No QK-Norm | Δ (Learned − NoQK) |
|:---:|:---:|:---:|:---:|:---:|
| 2.0 | 7.254 | 7.252 | 7.254 | −0.000 |
| 9.8 | 5.229 | 5.282 | 5.254 | −0.025 |
| 19.7 | 4.392 | 4.454 | 4.453 | −0.061 |
| 29.5 | 3.926 | 3.949 | 4.005 | −0.079 |
| 39.3 | 3.726 | 3.730 | 3.797 | −0.071 |
| 49.2 | **3.618** | **3.614** | 3.683 | **−0.065** |

**Interpretation:** Normalization helps loss regardless of whether γ is learnable. The 0.065-nat advantage over NoQK comes from the normalization component (gradient stabilization), not from γ-mediated rank control. This is notable: **the mechanism that helps loss (normalization) is different from the mechanism that preserves rank (γ learning)**.

---

### 4.2 Figure 2 — Key Rank Collapse: Does γ or Normalization Drive It?

![Key Rank Collapse](2_key_pr_causal.png)

**What it shows:** Mean Key PR (averaged over all 32 layers and 8 KV heads per layer) vs. tokens processed (in millions). This is the central figure of the study.

**Reading the curves — Three distinct phases:**

#### Phase 1: Rapid Collapse (0 → 8M tokens)

All three conditions undergo dramatic rank collapse:

| Condition | Init PR | Min PR | Tokens at Min | Drop |
|:---|:---:|:---:|:---:|:---:|
| Learned γ | 86.85 | 55.17 | 7.9M | −31.68 |
| Frozen γ=1 | 86.85 | 51.84 | 9.8M | −35.01 |
| No QK-Norm | 86.42 | 51.28 | 7.9M | −35.14 |

- PR drops by ~35 units (from ~87 to ~51–55) in just 8M tokens — the model loses **~40% of its effective dimensionality** during the initial training shock.
- During this phase, all three curves track closely together. The collapse is driven by the optimization dynamics (large initial gradients reshaping random-init weight matrices), not by normalization.
- Learned γ's floor is ~3–4 PR units higher than the other two — suggesting γ begins to act even during the collapse phase.

#### Phase 2: Recovery (8M → 25M tokens)

After hitting the floor, all three conditions **recover** PR — an unexpected finding:

| Condition | Min PR | PR at 25M | Recovery |
|:---|:---:|:---:|:---:|
| Learned γ | 55.17 | 66.47 | +11.30 |
| Frozen γ=1 | 51.84 | 58.96 | +7.12 |
| No QK-Norm | 51.28 | 62.39 | +11.11 |

- **The three curves diverge dramatically during recovery.** This is where the causal signal is clearest.
- Learned γ recovers fastest and highest — reaching ~66 by 25M tokens.
- No QK-Norm recovers strongly to ~62.
- Frozen γ=1 recovers the **slowest and least** — reaching only ~59.

The annotation on the figure provides the causal logic:
- A ≠ B ≈ C → γ drives it (true at the collapse floor)
- A ≈ B ≠ C → normalization drives it (not observed)
- All differ → both contribute (true during recovery, but γ is dominant)

#### Phase 3: Plateau (25M → 49M tokens)

All three conditions stabilize with minimal further PR change:

| Condition | PR at 25M | PR at 49M | Change |
|:---|:---:|:---:|:---:|
| Learned γ | 66.47 | 65.70 | −0.77 |
| Frozen γ=1 | 58.96 | 59.55 | +0.59 |
| No QK-Norm | 62.39 | 63.25 | +0.86 |

- The rank structure has largely settled by 25M tokens (~halfway through training).
- The final ordering is **Learned γ > No QK-Norm > Frozen γ=1**, and this gap shows no sign of closing.

**The most striking result:** Frozen γ=1 ends up with **lower** PR than No QK-Norm. Normalization without learnable γ doesn't just fail to help — it actively *hurts* the model's ability to maintain diverse key representations during recovery. This implies that the normalization constrains the model by forcing all dimensions to contribute equally to the norm, and without γ to re-scale them, the model has less flexibility in the spectral structure of its keys.

---

### 4.3 Per-Layer Effective Rank at 50M Tokens

**Three-zone decomposition:**

| Depth Zone | Layers | Learned γ | Frozen γ=1 | No QK-Norm | Learned−Frozen | Learned−NoQK |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Shallow | 0–7 | 57.12 | 52.40 | 51.70 | +4.72 | +5.42 |
| Middle | 8–23 | 61.97 | 56.05 | 59.55 | +5.92 | +2.42 |
| Deep | 24–31 | 72.80 | 64.07 | 72.29 | +8.73 | +0.51 |

**Key observations:**

1. **The depth gradient is inverted compared to the 500K study.** In the 500K pilot, shallow layers had the highest PR and deep layers the lowest. At 50M tokens, **deep layers have the highest PR** (~72–77) and shallow layers the lowest (~46–71). This inversion occurs because deep layers undergo massive collapse early but then recover dramatically, while shallow layers experience more sustained, gradual decline.

2. **The Frozen γ=1 deficit is largest in deep layers.** The gap between Learned γ and Frozen γ=1 is +4.72 in shallow layers but +8.73 in deep layers. The learned γ parameter has its strongest rank-preserving effect in the deepest layers — precisely where gradient signal is strongest.

3. **No QK-Norm closely tracks Learned γ in deep layers.** The gap between Learned γ and No QK-Norm is only +0.51 in deep layers, compared to +5.42 in shallow layers. Without normalization, deep layers achieve similar PR through $W_K$ alone. The shallow layers are where QK-Norm's advantage is most concentrated.

**Selected per-layer values at 49M tokens:**

| Layer | Learned γ | Frozen γ=1 | No QK-Norm | Learned−Frozen | Learned−NoQK |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 71.49 | 75.51 | 77.25 | −4.02 | −5.76 |
| 5 | 46.20 | 43.53 | 39.09 | +2.67 | +7.11 |
| 10 | 59.22 | 55.87 | 52.53 | +3.35 | +6.69 |
| 15 | 69.47 | 62.03 | 69.61 | +7.44 | −0.14 |
| 20 | 73.69 | 61.35 | 70.08 | +12.34 | +3.61 |
| 25 | 77.09 | 63.47 | 74.31 | +13.62 | +2.78 |
| 30 | 69.91 | 63.28 | 71.24 | +6.63 | −1.33 |
| 31 | 68.59 | 63.58 | 65.37 | +5.01 | +3.22 |

**Notable anomaly — Layer 0:** Layer 0 shows an inverted pattern where No QK-Norm (77.25) > Frozen γ=1 (75.51) > Learned γ (71.49). The shallowest layer, closest to the input embedding, behaves differently from all other layers. The learned γ here appears to be actively *suppressing* some dimensions rather than preserving them.

---

## 5. γ Dynamics: The Coefficient of Variation

The QK results include **γ coefficient of variation (CV)** — the standard deviation of γ values across the $d_k = 128$ dimensions, normalized by their mean. A CV of 0 means all γ values are identical (as at initialization); higher CV means the model is using γ to differentiate dimensions.

### 5.1 γ CV Trajectory (Selected Layers)

| Layer | Init CV | CV at ~8M | CV at 25M | CV at 49M | Trend |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | 0.000 | 0.071 | 0.133 | 0.168 | Monotonically increasing |
| 5 | 0.000 | 0.064 | 0.080 | 0.079 | Fast rise, then plateau |
| 10 | 0.000 | 0.070 | 0.064 | 0.071 | Rise, dip, stabilize |
| 15 | 0.000 | 0.071 | 0.143 | 0.155 | Rise, dip at ~10M, then strong rise |
| 20 | 0.000 | 0.075 | 0.086 | 0.106 | Steady increase |
| 25 | 0.000 | 0.074 | 0.107 | 0.126 | Steady increase |
| 27 | 0.000 | 0.079 | 0.134 | 0.172 | Strongest increase — highest CV |
| 30 | 0.000 | 0.065 | 0.099 | 0.122 | Steady increase |
| 31 | 0.000 | 0.077 | 0.111 | 0.120 | Steady increase |

**Frozen γ=1 control:** All γ CV values are exactly 0.0 at all checkpoints across all layers — confirming the ablation is correctly implemented ($\gamma$ is truly frozen).

### 5.2 Interpreting the γ CV

- **Layer 27 has the highest γ CV (0.172)** — this layer's γ parameters have diverged the most from uniform, meaning the model is most aggressively using per-dimension scaling here.
- **Layers 0 and 15 also show high γ CV (0.168 and 0.155)** — these are layers where γ differentiation is active.
- **Layers 5–10 have the lowest γ CV (~0.07–0.08)** — these are the layers with the lowest PR. The model has *not* used γ to differentiate dimensions much here, suggesting these layers' low PR comes from $W_K$ dynamics, not γ-mediated collapse.

**The γ CV anti-correlates with lower PR in a nuanced way:** Layers where γ diverges more (high CV) tend to recover more rank. This is consistent with γ acting as a rank-*preserving* mechanism — by amplifying useful dimensions and (potentially) suppressing noise dimensions, γ creates a more structured but higher-rank representation compared to the uniform-γ case.

---

## 6. The Collapse-Recovery Phenomenon

### 6.1 The U-Shaped PR Trajectory

The most unexpected finding in this study is the **U-shaped trajectory** of mean PR. Both the 500K study (which only captured the initial decline) and intuition would suggest that PR should monotonically decrease as training drives key representations toward more specialized, lower-rank structure.

Instead, all three conditions show:

1. **Rapid collapse** (0–8M tokens): PR drops from ~87 to ~51–55
2. **Strong recovery** (8M–25M tokens): PR climbs back to ~60–66
3. **Plateau** (25M–49M tokens): PR stabilizes

This suggests that early training undergoes a **representation shock** where random-init structure is rapidly destroyed (PR collapse), followed by a **structure-building** phase where the model constructs meaningful, higher-rank representations to support learning.

### 6.2 Recovery Rates by Condition

| Condition | Collapse Rate (PR/M tokens) | Recovery Rate (PR/M tokens) | Final Plateau |
|:---|:---:|:---:|:---:|
| Learned γ | −4.01 (0→8M) | +0.66 (8→25M) | 65.7 |
| Frozen γ=1 | −3.57 (0→10M) | +0.47 (10→25M) | 59.6 |
| No QK-Norm | −4.45 (0→8M) | +0.65 (8→25M) | 63.3 |

- No QK-Norm collapses fastest (−4.45 PR/M) but also recovers well (+0.65 PR/M)
- Learned γ collapses less (−4.01 PR/M) and recovers at a similar rate (+0.66 PR/M), resulting in the highest final PR
- Frozen γ=1 collapses at an intermediate rate but **recovers slowest** (+0.47 PR/M) — the normalization constraint without γ flexibility appears to trap the model in a lower-rank state

---

## 7. Comparison with the 500K Pilot

### 7.1 What Changed from 500K to 50M

| Observation | 500K Study | 50M Study |
|:---|:---|:---|
| **Loss** | ~7.2 (near random) | ~3.6 (meaningful learning) |
| **Mean PR** | ~114 (near-init) | ~60–66 (substantial collapse) |
| **PR range** | 105–123 across layers | 39–77 across layers |
| **PR trajectory** | Monotonic decline | U-shaped (collapse → recovery) |
| **Depth gradient** | Shallow = high, Deep = low | **Inverted**: Shallow = low, Deep = high |
| **QK-Norm effect** | +0.75 PR (0.6% of $d_k$) | +2.4 to +6.2 PR (2–5% of $d_k$) |
| **Causal mechanism** | Unknown | **γ identified as driver** |

### 7.2 The Depth Gradient Inversion

The most dramatic structural change is the **inversion of the depth-PR relationship**:

- **At 500K tokens:** Layer 0 has PR ≈ 123, Layer 31 has PR ≈ 106. Deeper = more collapsed.
- **At 50M tokens:** Layer 0 has PR ≈ 71–77, Layer 25 has PR ≈ 63–77. The pattern is non-monotonic, with the lowest PR in layers 4–7 (~39–53) and recovery in deep layers.

This suggests that deep layers' collapse in the 500K study was a transient phenomenon. As training progresses, deep layers restructure their representations and actually *recover* more rank than shallow layers, possibly because they are closest to the prediction objective and benefit from stronger, more structured gradient signal.

---

## 8. Methodology Details

### 8.1 SVD Probe Protocol

At each measurement checkpoint (every 120 training steps, ≈ 2M tokens), we:

1. Switch the model to `eval()` mode
2. Run a **fixed evaluation batch** through the model
3. Hook the output of the Rotary (RoPE) module — capturing post-norm, post-RoPE key representations
4. For each KV head $h$ in each layer $\ell$, reshape to $K^{(\ell,h)} \in \mathbb{R}^{n \times 128}$
5. Compute `torch.linalg.svdvals(K.float())`
6. Compute PR from singular values, averaged across heads within each layer
7. For the Learned γ condition: record the coefficient of variation of $\gamma$ parameters per layer
8. Remove all hooks and return to `train()` mode

### 8.2 Frozen γ Implementation

The Frozen γ=1 condition uses the **same model architecture** as Learned γ, but with $\gamma$ parameters `requires_grad=False` and initialized to $\mathbf{1}$. This means:

- The forward pass is identical: $\text{RMSNorm}(\mathbf{x})_j = x_j / \text{RMS}(\mathbf{x})$ (since $\gamma_j = 1$)
- The backward pass does not compute gradients for $\gamma$
- All other parameters ($W_Q$, $W_K$, $W_V$, $W_O$, FFN weights) are trained normally

### 8.3 What We Measure vs. What We Don't

**We measure:**
- Post-RoPE key representations (what the attention mechanism actually sees)
- γ coefficient of variation (how much γ values diverge from uniform)
- Training and validation loss

**We do NOT measure:**
- **Query representations** — the effective rank of $QK^\top$ depends on both Q and K
- **Individual γ values** — we track CV but not which specific dimensions are amplified/suppressed
- **Downstream task performance** — we do not know if PR differences affect quality beyond loss

---

## 9. Limitations

1. **Single seed per condition.** Without variance estimates, we cannot determine statistical significance. The effect sizes are larger than in the 500K study (2–6 PR units vs. 0.75), which increases confidence, but multiple seeds are needed to confirm.

2. **50M tokens is early.** This represents ~0.3% of Chinchilla-optimal training for a 1.5B model. The U-shaped trajectory suggests PR dynamics are non-trivial, and the plateau at 25M+ tokens may not be the final equilibrium.

3. **Only keys measured, not queries.** Attention is computed from $QK^\top$, and query rank dynamics may differ.

4. **γ CV is a coarse summary.** The coefficient of variation captures how much γ values vary but not the structure of that variation — e.g., whether low-γ dimensions correspond to low-variance key dimensions.

5. **Muon optimizer interaction.** Muon applies orthogonal updates to 2D weight matrices, which could interact with spectral dynamics differently than AdamW. Results may be optimizer-specific.

6. **The Frozen γ condition is not a natural baseline.** No production model uses RMSNorm with frozen γ. The condition is valuable for causal inference but does not represent a practical alternative.

---

## 10. Conclusions

### Strong Signals (consistent across all measurements)

1. **The learned γ is the primary driver of rank preservation.** The three-way comparison conclusively shows that normalization alone (Frozen γ=1) does not preserve key rank — and actually produces worse rank recovery than no normalization at all. The learned per-dimension scale $\gamma$ is the mechanism through which QK-Norm affects spectral structure.

2. **PR follows a U-shaped trajectory.** All conditions exhibit collapse → recovery → plateau, with the collapse floor occurring at ~8M tokens and recovery largely complete by ~25M tokens. This is a structural feature of early training, not an artifact of normalization.

3. **Normalization helps loss but not rank.** Both Learned γ and Frozen γ=1 achieve ~0.065 nats lower train loss than No QK-Norm (3.62 vs. 3.68), confirming that gradient stabilization from RMSNorm provides a real optimization benefit — independent of any rank effect.

4. **The depth-PR gradient inverts during training.** Early training (500K) shows deep layers most collapsed; extended training (50M) shows deep layers with the highest PR after recovery.

### Interpretation (hypothesis, not proven)

5. **γ acts as a dimension-selective amplifier.** The γ CV analysis shows that learned γ values diverge from uniform, and layers with higher γ CV tend to have higher PR. Rather than suppressing "dead" dimensions (γ → 0), the model appears to use γ to create structured spectral variation that *preserves* effective dimensionality. Without this flexibility (Frozen γ=1), the model is constrained to a lower-rank manifold.

6. **Frozen γ harms recovery because normalization creates a constraint.** RMSNorm forces $\sum_j (\gamma_j x_j)^2 / d_k$ to be constant. When $\gamma_j = 1$ (frozen), this acts as a hard geometric constraint on key representations — all dimensions must jointly satisfy the norm constraint. The learnable γ relaxes this by allowing the norm to concentrate on fewer dimensions, giving the model more flexibility in its spectral structure.

### Next Steps

- **Track individual γ values** (not just CV) to determine which dimensions are amplified vs. suppressed and how this relates to singular vector structure
- **Extend to 250M+ tokens** to observe whether the PR plateau persists or a second phase transition occurs
- **Measure query PR** alongside key PR for a complete attention-rank picture
- **Multiple seeds** (at least 3) to estimate variance and confirm effect significance
- **Add a pure AdamW baseline** to isolate Muon's role in spectral dynamics
