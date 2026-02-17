# Experimental Plan: The Dynamics of Dimensional Collapse
## What Drives Rank Collapse — Normalization, Optimizer, or Data Signal?

---

## 1. The Problem With The Current Experiment

Our previous experiment had three fatal confounds:

| Issue | Why It Invalidates Results |
|---|---|
| **Only measured QK-norm gamma** | We proved gamma is sparse, not that the underlying representations are low-rank. These are different claims. |
| **2M training tokens** | 44:1 param-to-data ratio. The model is data-starved — of course it collapses. |
| **No controls** | No comparison without QK-norm, no comparison with different optimizers, no ablation on data volume. We can't attribute cause. |
| **Static snapshot** | We measured the end-state but not the trajectory. When does collapse happen? Is it sudden or gradual? |

**What IS already known (and we should NOT re-discover):**
- Attention outputs are inherently low-rank (Dong et al. 2021, Bhojanapalli et al. 2020)
- You can prune 30-60% of attention heads without much damage (Michel et al. 2019)
- Self-attention without skip connections loses rank doubly-exponentially with depth
- Structured pruning works on trained models (SparseGPT, Wanda)

---

## 2. The Novel Questions

Three questions that are **genuinely under-explored**, especially at this scale with full training access:

### Q1: Is QK-Norm the *cause* or just a *reporter* of collapse?
> **Hypothesis:** QK-Norm (RMSNorm on K) doesn't cause the rank to collapse — it just makes the collapse *visible* through its gamma vector. The underlying W_k matrix is equally low-rank without QK-norm, but you can't see it as easily.
>
> **Alternative Hypothesis:** QK-Norm *accelerates* collapse by providing a cheap mechanism (zero-out gamma) vs. the harder path of rotating W_k to be low-rank.
>
> **Why this matters:** If QK-norm accelerates collapse, it has implications for every modern LLM using it (Gemma, etc.). If it just reports collapse, then the gamma vector is a free diagnostic tool.

### Q2: Does the Muon optimizer change collapse dynamics vs. AdamW?
> **Hypothesis:** Muon (which applies Newton-like spectral updates to 2D weight matrices) may produce a *different collapse pattern* than pure AdamW, because it explicitly shapes the spectral structure of weights.
>
> **Why this matters:** Nobody has studied how optimizer choice affects the spectral structure of attention weight matrices during training. This is tractable at our scale and would be a genuine contribution.

### Q3: Does collapse depth scale with data complexity, not just volume?
> **Hypothesis:** The *effective rank at convergence* is determined by the intrinsic dimensionality of the data distribution, not just by parameter count. Training on more complex data (code, math) will produce higher effective ranks than training on simple data (children's stories), even with the same token budget.
>
> **Why this matters:** This would prove collapse is *adaptive* (the model right-sizes itself to the task), not *pathological* (the model is broken). This distinction drives whether collapse should be prevented or exploited.

---

## 3. Experimental Design

### 3.1 — The Proper Measurement (Replace Gamma-Watching)

**The current method is wrong.** Measuring `k_norm.weight` only tells you about the normalization layer's learned scaling. We need to measure the *actual* rank of the key representations.

**Correct metrics to compute (at checkpointed steps during training):**

```
For each layer L, for each attention head H:
  1. Collect K = W_k(X) for a fixed evaluation batch (1000 tokens)
  2. Compute SVD: K = UΣV^T
  3. Record:
     a. Singular value spectrum: σ_1, σ_2, ..., σ_dk
     b. Participation Ratio: PR = (Σ σ_i)^2 / Σ(σ_i^2)
     c. 90%-energy rank: min r such that (Σ_{i=1}^r σ_i^2) / (Σ σ_i^2) >= 0.90
     d. Condition number: σ_1 / σ_dk

  Also compute the same for Q representations.
```

**Additionally, keep gamma tracking as a secondary metric** — but the primary claim must be about the actual representations, not the normalization weights.

### 3.2 — Controlled Experiment Matrix

Run **6 training runs** in a 3×2 factorial design:

| Run | QK-Norm | Optimizer | Tokens | Purpose |
|-----|---------|-----------|--------|---------|
| A1  | ✅ Yes  | Muon+AdamW (current) | 25M | **Baseline** (current setup, more data) |
| A2  | ❌ No   | Muon+AdamW | 25M | **Isolate QK-norm effect** |
| B1  | ✅ Yes  | Pure AdamW | 25M | **Isolate Muon effect** |
| B2  | ❌ No   | Pure AdamW | 25M | **Double control** |
| C1  | ✅ Yes  | Muon+AdamW | 25M, simple data | **Data complexity test** |
| C2  | ✅ Yes  | Muon+AdamW | 25M, complex data | **Data complexity test** |

**Why 25M tokens:** At our model size (88M params), 25M tokens gives a ~3.5:1 ratio. This is still small by industry standards, but it's in the regime where the model actually starts learning meaningful representations (we already know from our 8M runs that it reaches ~4.0 val loss). 25M is feasible on a single GPU in a few hours.

**Data splits for C1/C2:**
- **C1 (simple):** Filter cosmopedia for short, simple-vocabulary documents (stories, basic explanations)
- **C2 (complex):** Filter cosmopedia for technical/scientific documents with richer vocabulary and structure

### 3.3 — Measurement Schedule

Measure the SVD metrics **during training**, not just at the end:

```
Checkpoints at: 0, 500K, 1M, 2M, 4M, 8M, 12M, 16M, 20M, 25M tokens
```

This gives us the **trajectory** of collapse — when it starts, how fast it progresses, whether it stabilizes.

**Fixed evaluation batch:** Use the **exact same 1000-token batch** for all SVD measurements across all runs. Sample it once, save it to disk, reuse everywhere. This eliminates data-dependent noise in the measurements.

### 3.4 — The "Surgery" Test (Done Right)

Redo the causal ablation, but properly:

1. **Prune based on SVD, not gamma:** Zero out the bottom singular vectors of the actual K matrix (via W_k modification), not the gamma weights.
2. **Compare pruning strategies:**
   - Strategy A: Prune by gamma magnitude (current method)
   - Strategy B: Prune by singular value magnitude (SVD-based)  
   - Strategy C: Random pruning (control)
3. **Measure on a held-out test set** that was NOT used for any SVD computation.

If Strategy A ≈ Strategy B >> Strategy C, then gamma IS a reliable proxy for true rank (which itself would be a useful finding). If Strategy B >> Strategy A, then gamma is misleading.

---

## 4. What Each Comparison Tells Us

### Comparison 1: A1 vs A2 → "Does QK-Norm cause or report collapse?"
- If A1 and A2 have **similar SVD rank profiles** → QK-Norm just reports collapse
- If A1 has **lower rank** than A2 → QK-Norm accelerates collapse
- If A1 has **higher rank** than A2 → QK-Norm somehow helps (unlikely but would be very interesting)

### Comparison 2: A1 vs B1 → "Does Muon change spectral structure?"
- If A1 and B1 have **different collapse patterns** → Muon shapes rank evolution (novel finding about optimizer-architecture interaction)
- If same → collapse is architecture-determined, optimizer-independent

### Comparison 3: C1 vs C2 → "Is collapse data-adaptive?"
- If C2 has **higher effective rank** → the model right-sizes to data complexity (collapse is adaptive, not broken)
- If C1 ≈ C2 → collapse is architecture-determined, data-independent (more concerning for efficiency claims)

### Comparison 4: A1 across time → "When does collapse happen?"
- Plot the trajectory of effective rank vs. training step
- Look for phase transitions (sudden drops) vs. gradual decay
- Correlate with loss curve — does rank collapse coincide with loss plateaus?

---

## 5. Expected Outputs

### 5.1 — Figures

1. **The Trajectory Plot** (signature figure): 
   - X-axis = training tokens, Y-axis = effective rank (participation ratio)
   - 4 lines: A1, A2, B1, B2 (all on same plot)
   - This single figure tells the whole story of Q1 and Q2

2. **Singular Value Waterfall**:
   - For runs A1 and A2 at final checkpoint
   - X-axis = singular value index, Y-axis = σ_i / σ_1 (normalized)
   - Shows the actual spectral shape, not just a summary statistic

3. **Data Complexity Comparison**:
   - Side-by-side: PR for C1 vs C2, per layer
   - Shows whether more complex data = higher rank

4. **Surgery Comparison**:
   - X-axis = % pruned, Y-axis = val loss
   - 3 lines: SVD-based, Gamma-based, Random
   - Validates whether gamma is a reliable proxy

5. **Gamma vs SVD Correlation Scatter**:
   - X = gamma magnitude for each dimension
   - Y = corresponding singular value contribution
   - Shows the relationship (or lack thereof) between gamma and true importance

### 5.2 — Claims We Can Make

If results go as hypothesized, the paper narrative becomes:

> "We show that dimensional collapse in small-scale Transformers is jointly determined by 
> normalization architecture, optimizer choice, and data complexity. QK-Norm [accelerates/does not affect] 
> collapse compared to un-normalized baselines. We further demonstrate that the Muon optimizer produces a 
> [different/similar] spectral evolution compared to AdamW, suggesting that [optimizers/architecture] 
> [is/is not] the primary driver of rank collapse. Finally, we provide evidence that collapse depth is 
> [adaptive to/independent of] data complexity, with implications for structured pruning strategies."

This is a **mechanistic, controlled study** — not a re-discovery of a known phenomenon.

---

## 6. Implementation Checklist

### Phase 0: Data Preparation (~15 min)
- [ ] Prepare 25M token dataset from cosmopedia (standard)
- [ ] Prepare "simple" subset (~25M tokens, filtered by vocabulary complexity / sentence length)  
- [ ] Prepare "complex" subset (~25M tokens, filtered for technical content)
- [ ] Extract and save a fixed 1000-token evaluation batch to disk

### Phase 1: Measurement Infrastructure (~1 hour coding)
- [ ] Write `svd_probe.py` — hooks into model, computes SVD metrics at each checkpoint
- [ ] Write `checkpoint_callback.py` — saves model state at the 10 measurement points
- [ ] Ensure SVD computation uses float32 (not bfloat16) for numerical stability
- [ ] Validate SVD metrics on a random model (sanity check: should have near-full rank)

### Phase 2: Model Variant (No QK-Norm) (~30 min coding)
- [ ] Create `MultiHeadAttention_NoQKNorm` variant (remove q_norm and k_norm)
- [ ] Verify it trains stably without QK-norm (may need LR adjustment)
- [ ] Create `LLMConfig` variant with flag `use_qk_norm: bool = True`

### Phase 3: Optimizer Variant (Pure AdamW) (~15 min coding)
- [ ] Create `setup_pure_adamw_optimizer()` that puts ALL params on AdamW
- [ ] Match total learning rate budget (use `adamw_lr` for everything)

### Phase 4: Run Experiments (~6 runs × ~2-3 hours each = 12-18 hours GPU)
- [ ] Run A1 (baseline with QK-norm + Muon, 25M tokens)
- [ ] Run A2 (no QK-norm + Muon, 25M tokens)
- [ ] Run B1 (QK-norm + pure AdamW, 25M tokens)
- [ ] Run B2 (no QK-norm + pure AdamW, 25M tokens)
- [ ] Run C1 (QK-norm + Muon, 25M simple data)
- [ ] Run C2 (QK-norm + Muon, 25M complex data)

### Phase 5: Analysis & Plotting (~2 hours)
- [ ] Generate all 5 figures
- [ ] Compute statistical significance where applicable
- [ ] Run the 3-way surgery comparison
- [ ] Write final report with controlled claims

---

## 7. What NOT To Do

| Temptation | Why To Avoid It |
|---|---|
| Train for only 2M tokens | Data starvation dominates everything, making other effects invisible |
| Only measure gamma weights | Confuses the normalization layer's behavior with the model's behavior |
| Claim this is "never been seen before" | Rank collapse has been studied since 2019+ |
| Conclude that collapse = wasted compute | The model needs the full width to explore during training |
| Run one config and draw conclusions | Without controls, you can't attribute causation |
| Use bfloat16 for SVD computation | Numerical errors in SVD with half-precision will corrupt your metrics |

---

## 8. Realistic Novelty Assessment

| Claim | Novelty Level | Why |
|---|---|---|
| "Rank collapse exists in attention" | ❌ None | Dong et al. 2021, Bhojanapalli et al. 2020 |
| "QK-Norm gamma becomes sparse" | ❌ None | Expected behavior of learnable normalization |
| "You can prune dimensions" | ❌ None | Michel et al. 2019, SparseGPT, Wanda |
| "QK-Norm accelerates/reports collapse" | ✅ Moderate | Nobody has isolated this with controlled ablation |
| "Muon changes spectral evolution" | ✅ High | No prior work on Muon + spectral dynamics |
| "Collapse rate scales with data complexity" | ✅ Moderate | Observed but not systematically studied at this scale |
| "Gamma magnitude predicts SVD importance" | ✅ Moderate | Practical finding for pruning without expensive SVD |
| "Phase transition in rank during training" | ✅ High if found | Training dynamics of collapse trajectory is understudied |

**The strongest potential finding is Q2 (Muon effect)** — because Muon is new enough that nobody has studied its spectral impact on attention weights, and you have the full codebase to ablate it cleanly.
