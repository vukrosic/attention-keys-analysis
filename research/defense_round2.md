# Defense Against Critique — Round 2

## Triage

| Critique Point | Classification |
|:---|:---|
| #1: PR(diag(γ)) is a category error | **Valid concern, but partially a misunderstanding** |
| #2: RMSNorm doesn't work the way you think | **Valid technical point — accept and adapt** |
| #3: You're measuring the wrong thing | **Misunderstanding of the experimental role** |
| #4: No correlation between γ and singular vectors | **Valid — accept and implement** |
| #5: 25M tokens too early | **Valid — accept and adjust** |
| #6: Can't compare γ across conditions | **Misunderstanding — this is by design** |
| #7: Full γ vectors lost on crash | **Valid — trivial fix** |

---

## Point-by-Point Defense

### Critique #1: "PR(diag(γ)) is mathematically meaningless"

**The Defense:**

The critic is correct that γ entries are not singular values, and that calling the metric "PR(diag(γ))" invites a false equivalence. But the critic then overstates the problem.

The quantity $(Σ|γ_j|)² / Σ|γ_j|²$ is the **inverse Herfindahl index** of the γ magnitudes — a well-established concentration measure used in economics, ecology, and information theory. It answers a perfectly well-defined question: *"How many dimensions does γ effectively leave alive?"* If γ = [1, 1, ..., 1, 0, 0, ..., 0] with k nonzero entries, this metric returns exactly k. That's useful.

The critic is right that this metric can miss more subtle sculpting (e.g., γ = [1.0, 0.3, 0.3, ...] has high "γ-PR" but still concentrates the spectrum). But that's a limitation of the metric's sensitivity, not a mathematical error.

**The Solution:**

Rename this metric from `gamma_pr` to `gamma_herfindahl` or `gamma_effective_dims` to avoid the false equivalence with Key PR. Add **γ coefficient of variation** (`std/mean`) as the primary γ metric, since it captures non-uniformity even when no dimensions are near zero. Plot both.

**The Trade-off:** Two metrics instead of one — minor added complexity, but more honest.

---

### Critique #2: "RMSNorm doesn't work the way you think — γ-std matters, not γ-mean"

**The Defense:**

This is correct. The critic identified the real signal: it's the **non-uniformity** of γ (its std or CV) that drives PR change, not its mean. If all γ_j move together (e.g., all decay from 1.0 to 0.8), RMSNorm's output PR is unchanged.

We already compute γ-std at every checkpoint. The issue was that we weren't foregrounding it.

**The Solution:**

Make `γ_cv = γ_std / γ_mean` the headline panel for γ analysis. Replace the γ mean±std evolution plot (Panel G) with a γ-CV evolution plot per layer. This directly measures the quantity that affects PR.

Additionally, plot per-layer γ-CV vs per-layer ΔPR(QK−NoQK) at the final checkpoint as a scatter plot. If the correlation is positive (layers where γ is more non-uniform have larger QK-Norm effect on PR), that's the evidence.

**The Trade-off:** None. We already have the data. This is a better visualization of the same measurements.

---

### Critique #3: "You're measuring the wrong thing for the thesis. The direct test is Panel B."

**The Defense:**

The critic says "Panel B (Key PR trajectory for QK vs NoQK) IS the answer." This misses the point. Panel B tells us *whether* QK-Norm changes PR dynamics. It does NOT tell us *why*. 

The thesis isn't "does QK-Norm affect PR" — we already know that from the 500K pilot. The thesis is "does γ *explain* the difference." Panel B is necessary but not sufficient.

The critic then says the only way to establish causality is a frozen-γ control. That's ideal but not the only path. We can establish a strong *correlational* case without it:

1. If γ-CV increases over training → γ is learning non-uniform scalings
2. If γ-CV at final checkpoint correlates with ΔPR across layers → γ non-uniformity predicts the QK-Norm effect
3. If the dimensions γ suppresses align with the trailing singular vectors → γ is actively steering the collapse

These three pieces together constitute a strong *circumstantial* case. The frozen-γ experiment would upgrade it to *causal*. We should state this distinction explicitly.

**The Solution:**

Add the frozen-γ run. It's cheap — literally one line: `block.attention.k_norm.weight.requires_grad_(False)` and `block.attention.q_norm.weight.requires_grad_(False)` for each block. This creates a third condition:

- **QK-Norm (learned γ)**: Full QK-Norm as designed
- **QK-Norm (frozen γ=1)**: RMS normalization but no learned scaling  
- **No QK-Norm**: Identity

This is only ~30 extra minutes of H100 time. It disentangles: is it the normalization (dividing by RMS) or the learned scaling (γ) that drives the PR difference?

**The Trade-off:** One extra run (~30 min). The overnight budget comfortably absorbs this.

---

### Critique #4: "You never correlate which dimensions γ suppresses with which singular vectors have low values"

**The Defense:**

This is a valid and excellent suggestion. We already save the full spectrum (`s.tolist()`) and the full γ vector. We just never compute the correlation.

However, the analysis is slightly more nuanced than the critic suggests. The singular vectors of K depend on both the input distribution AND W_K AND γ (if QK-Norm). So asking "does γ suppress dimensions that align with trailing singular vectors" is asking whether γ is *reinforcing* a collapse that's already happening in the data/weight space, or *causing* it independently.

**The Solution:**

At the final checkpoint, for each layer:
1. Compute SVD of K: $K = UΣV^T$
2. Take the right singular vectors $V$ (shape: [d_k, d_k])
3. Project γ into the singular basis: $\gamma_{\text{spectral}} = V^T \gamma$
4. Compute the Spearman rank correlation between $\gamma_{\text{spectral},j}$ and $\sigma_j$

If this correlation is positive (γ amplifies dominant directions, suppresses trailing ones), γ is *reinforcing* the existing spectral structure. If negative, γ is *opposing* it. If near zero, γ is spectral-structure-agnostic.

This requires saving V at the final checkpoint (not just the singular values), which means changing `compute_rank_metrics` to return the right singular vectors. Minor code change.

**The Trade-off:** Slightly larger final JSON (128×128 matrix per head per layer at final checkpoint only). Negligible.

---

### Critique #5: "25M tokens is probably still too early"

**The Defense:**

Agree. The critic is right that 1,526 optimizer steps may not be enough for γ to differentiate substantially.

**The Solution:**

Change `TARGET_TOKENS` to 50M (default) with a note to extend to 100M if training speed allows. 50M doubles the optimizer steps to ~3,000, giving γ more time to specialize. We have the overnight budget — there's no reason not to use it.

If the H100 completes 50M × 2 runs + frozen-γ run in 3 hours (very likely), and we still have 5+ hours of compute, extend the QK-Norm run to 100M to observe the full trajectory.

**The Trade-off:** More training time, but we have it.

---

### Critique #6: "The NoQK-Norm model has no γ. You can't compare γ-PR across conditions."

**The Defense:**

This is a misunderstanding. We don't compare γ-PR across conditions. The γ analysis is **internal to the QK-Norm model** — we're asking whether the *pattern* of γ specialization within the QK-Norm model explains why its PR trajectory differs from the NoQK model.

The comparison structure is:
- **Panel B**: QK PR vs NoQK PR → establishes that there IS a difference
- **Panels G/G2**: γ dynamics within QK model → investigates the *mechanism* of that difference
- **Frozen-γ control** (new): QK-frozen-γ PR vs QK-learned-γ PR → isolates γ's causal contribution

The critic's concern about comparing γ across conditions is a strawman — that was never the intent.

**The Solution:** No code change needed, but the report should explicitly state: "γ analysis applies only to the QK-Norm condition. Cross-condition comparison uses Key PR as the shared metric."

---

### Critique #7: "Full γ vectors lost on crash"

**The Defense:**

Valid. Trivial fix.

**The Solution:**

Save full γ vectors in the last 5 intermediate saves, not just the final checkpoint. Add a small ring buffer that keeps the last 5 checkpoints' full γ values.

**The Trade-off:** ~5 × 32 × 128 × 4 bytes = ~80KB extra per intermediate save. Negligible.

---

## Revised Plan (V1.1)

### Changes from V1.0

1. **Add frozen-γ control run** (third condition: QK-Norm with `weight.requires_grad_(False)`)
2. **TARGET_TOKENS = 50M** (default), extend to 100M if time allows
3. **Replace γ-PR with γ-CV** as the primary γ non-uniformity metric
4. **Add γ-spectral correlation** (Spearman between γ projected into singular basis and σ values) at final checkpoint
5. **Save full γ vectors** in last 5 intermediate saves for crash resilience
6. **Rename** `gamma_pr` to `gamma_effective_dims` to avoid false equivalence with Key PR

### Thesis (refined)

**Does the *learned non-uniformity* of QK-Norm's γ parameter drive the PR difference between QK-Norm and non-QK-Norm models, or is the difference caused by the RMS normalization itself?**

### Three conditions, one answer

| Condition | What it isolates |
|:---|:---|
| QK-Norm (learned γ) | Full effect: normalization + learned scaling |
| QK-Norm (frozen γ=1) | Normalization effect only |
| No QK-Norm | Baseline |

If `PR(QK-learned) ≠ PR(QK-frozen) ≈ PR(NoQK)` → γ matters, normalization doesn't.
If `PR(QK-learned) ≈ PR(QK-frozen) ≠ PR(NoQK)` → normalization matters, γ doesn't.
If all three differ → both contribute.
