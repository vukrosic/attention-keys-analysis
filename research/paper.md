# Rank Collapse and Recovery in Attention Keys: How Learned γ in QK-Norm Controls Dimensional Structure at 1.5B Scale

---

## Key Findings at a Glance

This study trains a **1.5B parameter** LLM under three conditions to answer one question: *when QK-Norm helps attention heads, is it because of the normalization itself or because of the learned γ weights?* Below we first explain every concept you need to understand the results, then walk through the figures step by step.

---

### Concepts You Need First

#### What is γ (gamma)?

In a transformer attention head, each token is projected into a 128-dimensional **key** vector. **QK-Norm** applies RMSNorm to this vector, which does two things:

1. **Normalize** — divide every element by the root-mean-square of the whole vector, so the vector has a controlled magnitude.
2. **Scale by γ** — multiply each of the 128 elements by its own learnable weight $\gamma_j$.

$$\text{RMSNorm}(\mathbf{x})_j = \gamma_j \cdot \frac{x_j}{\sqrt{\frac{1}{128}\sum_{i=1}^{128} x_i^2}}$$

Think of γ as a **per-dimension volume knob**. Each of the 128 dimensions gets its own knob, initialized to 1.0 (all equal). As training progresses, the model can turn some knobs up (amplify that dimension) and others down (suppress it), giving it direct control over which dimensions matter.

**Concrete example** with a 4-dimensional vector $\mathbf{x} = [2, 4, 6, 8]$:
- RMS = $\sqrt{(4 + 16 + 36 + 64)/4} = \sqrt{30} \approx 5.48$
- After normalization: $[0.365,\ 0.730,\ 1.095,\ 1.461]$ (all values rescaled)
- Now suppose training learns $\gamma = [1.5, 1.0, 0.5, 0.1]$:
  1. Dim 1: $0.365 × 1.5 = 0.548$ ← **amplified**
  2. Dim 2: $0.730 × 1.0 = 0.730$ ← unchanged
  3. Dim 3: $1.095 × 0.5 = 0.548$ ← **dampened**
  4. Dim 4: $1.461 × 0.1 = 0.146$ ← **nearly removed**
- Final: $[0.55,\ 0.73,\ 0.55,\ 0.15]$ — γ has reshaped which dimensions carry information.

#### What is Participation Ratio (PR)?

PR measures **how many of the 128 dimensions are actually being used** in the key vectors. It answers: "is the model spreading information across many dimensions, or cramming everything into a few?"

- **PR = 128** → all dimensions contribute equally (maximum diversity, full rank)
- **PR = 1** → only one dimension carries any signal (total collapse)
- **PR ≈ 60** (what we observe) → roughly 60 of 128 dimensions are meaningfully active

When PR drops, it means the model is "wasting" dimensions — many key dimensions become redundant or near-zero. This is called **rank collapse** and can limit the model's representational capacity.

#### The Three Experimental Conditions

We train the same 1.5B model three times, changing only how QK-Norm is configured:

| Condition | Normalization? | γ learned? | Purpose |
|:---|:---:|:---:|:---|
| **Learned γ** | ✅ Yes | ✅ Yes | Full QK-Norm — the real thing |
| **Frozen γ=1** | ✅ Yes | ❌ No (locked at 1.0) | Ablation — has normalization but *not* the learned scaling |
| **No QK-Norm** | ❌ No | ❌ No | Baseline — no normalization at all |

**Why three conditions?** Because QK-Norm does two things at once (normalize + scale by γ). With only two conditions (on/off) we can't tell which part matters. The Frozen γ=1 condition surgically removes γ learning while keeping normalization, letting us isolate the cause:
- If Learned γ ≠ Frozen γ ≈ No QK-Norm → **γ is the driver** (removing γ makes it behave like no norm)
- If Learned γ ≈ Frozen γ ≠ No QK-Norm → **Normalization is the driver** (γ doesn't matter)
- If all three differ → **Both contribute**

---

### Figure 1: The Central Result — Does γ or Normalization Drive Rank?

![Key Rank Collapse — Does γ or Normalization Drive It?](./images/2_key_pr_causal.png)

**What this figure shows:** The y-axis is *Participation Ratio (PR)* — the number of effectively used dimensions (out of 128). Higher = healthier. The x-axis is training progress in millions of tokens.

**Reading it step by step:**

1. **All three models start at PR ≈ 87** (top-left). At random initialization, each of the 128 dimensions contributes roughly equally — the model hasn't learned anything yet.

2. **All three crash to PR ≈ 51–55 by ~8M tokens** (bottom of the U-curve). This is *rank collapse* — the model destroys its initial random structure as it begins learning. This happens regardless of whether QK-Norm is used. **Conclusion: normalization alone does NOT prevent collapse.**

3. **After the collapse floor, the three lines diverge** — this is where the experiment reveals its answer:
   - 🟠 **Learned γ (orange)** recovers the fastest and highest, reaching PR = **65.7**
   - 🔵 **No QK-Norm (cyan)** recovers to PR = **63.3** — surprisingly, the second-best
   - 🟣 **Frozen γ=1 (purple)** recovers the least, only to PR = **59.6** — the *worst* of all three

4. **The punchline:** If normalization were the key mechanism, the two normalized models (Learned γ and Frozen γ) would recover similarly. Instead, removing γ learning (Frozen) makes it *worse* than having no normalization at all. **Therefore, the learned γ parameter — not the normalization — is what drives rank recovery.**

### Figure 2: Loss Tells a Different Story

![Training & Validation Loss](./images/1_loss.png)

**What this figure shows:** Training and validation loss (lower = better language modeling) over the same 50M tokens.

**Reading it step by step:**

1. **Both normalized models (Learned γ and Frozen γ) achieve nearly identical loss** — around 3.62 for training, 3.44 for validation. They overlap almost perfectly.

2. **The No QK-Norm model has noticeably higher loss** — 3.68 training, 3.51 validation. Normalization clearly helps the model learn better.

3. **But here's the key insight:** Compare this to Figure 1 above. For *loss*, normalization helps equally whether γ is learned or frozen. For *rank*, only learned γ helps — frozen γ actually hurts. **This means loss and rank are driven by different mechanisms:**
   - **Normalization → stabilizes gradients → lowers loss** (doesn't need γ)
   - **Learned γ → selectively amplifies/suppresses dimensions → preserves rank** (needs γ)

### The Four Conclusions, Explained

> **Conclusion 1: "γ drives rank, normalization drives loss."**
>
> *What this means:* QK-Norm does two helpful things, but they are independent. The normalization step (dividing by RMS) helps the model train faster (lower loss). The learned γ weights help the model maintain a richer internal representation (higher rank). You need both for full benefit.

> **Conclusion 2: "PR follows a U-shaped trajectory."**
>
> *What this means:* Every model — with or without QK-Norm — goes through the same three phases: (1) rank collapses as random initialization is destroyed, (2) rank recovers as the model learns meaningful structure, (3) rank plateaus. This U-shape appears to be a fundamental feature of how transformers learn, not an artifact of any particular technique.

> **Conclusion 3: "Normalization without γ is counterproductive for rank."**
>
> *What this means:* This is the most surprising finding. If you apply QK-Norm but freeze γ at 1 (so every dimension is treated equally), you actually get *worse* rank than not using QK-Norm at all. Why? The normalization projects all key vectors onto a sphere, which constrains the geometry. Without γ to selectively stretch dimensions, this constraint *limits* the model's ability to differentiate dimensions, causing more collapse than if you just left the keys unnormalized.

> **Conclusion 4: "The depth-PR gradient inverts."**
>
> *What this means:* In a pilot study at 500K tokens, early layers had higher rank than deep layers. By 50M tokens, this reverses — deeper layers develop higher rank. The model's internal structure reorganizes as training progresses, with later layers developing more complex representations over time.

---

## 1. Overview

This study investigates the causal mechanism of **QK-Normalization** (RMSNorm applied to both query and key projections) on dimensional collapse in transformer attention heads. In our Gemma-style architecture, RMSNorm is applied to **both Q and K** independently before rotary position encoding — hence the name "QK-Norm."

We train a **1.5B parameter** dense LLM for **~49M tokens** under three conditions to disentangle the effect of normalization from the effect of the learned scale parameter ($\gamma$).

### What is γ?

When RMSNorm is applied to a vector, it first normalizes it (divides by root-mean-square), then re-scales each element by a learned weight. That learned weight vector is called $\gamma$ (gamma). Formally, for a vector $\mathbf{x} = [x_1, x_2, \ldots, x_{128}]$ (one dimension per entry of the attention head):

$$\text{RMSNorm}(\mathbf{x})_j = \gamma_j \cdot \frac{x_j}{\sqrt{\frac{1}{128}\sum_{i=1}^{128} x_i^2}}$$

Here:
- $j$ is the index of one specific dimension (1 through 128, since our head dimension $d_k = 128$)
- $x_j$ is the raw value at dimension $j$
- The denominator $\sqrt{\frac{1}{128}\sum_{i=1}^{128} x_i^2}$ is the **root-mean-square (RMS)** of the entire vector — a single scalar that measures the overall magnitude
- $\gamma_j$ is the learnable weight for dimension $j$ — initialized to 1.0 at the start of training

**Example:** Suppose we have a 4-dimensional vector $\mathbf{x} = [2, 4, 6, 8]$ with $\gamma = [1, 1, 1, 1]$:
- RMS = $\sqrt{(4 + 16 + 36 + 64)/4} = \sqrt{30} \approx 5.48$
- Normalized: $[2/5.48,\ 4/5.48,\ 6/5.48,\ 8/5.48] \approx [0.365,\ 0.730,\ 1.095,\ 1.461]$

**Step-by-step scaling with learned $\gamma$:**
Suppose training learns $\gamma = [1.5, 1.0, 0.5, 0.1]$ to prioritize the first dimension and suppress the last:
1. Dimension 1: $0.365 \times \mathbf{1.5} = 0.548$ (amplified)
2. Dimension 2: $0.730 \times \mathbf{1.0} = 0.730$ (unchanged)
3. Dimension 3: $1.095 \times \mathbf{0.5} = 0.548$ (dampened)
4. Dimension 4: $1.461 \times \mathbf{0.1} = 0.146$ (nearly removed)

Final output: $[0.55,\ 0.73,\ 0.55,\ 0.15]$ — **This per-dimension control is the key mechanism we study.**

Each layer has its own separate $\gamma$ vector for Q and another for K (so 32 layers × 2 = 64 separate $\gamma$ vectors across the model, each with 128 learnable values).

### The Three Conditions

1. **QK-Norm (Learned $\gamma$)**: Full method. RMSNorm + learnable per-dimension scaling ($\gamma$ is trained via backpropagation).
2. **QK-Norm (Frozen $\gamma=1$)**: Ablation. RMSNorm is applied, but $\gamma$ is locked at all-ones — the model gets normalization but cannot learn per-dimension scaling.
3. **No QK-Norm**: Baseline. No normalization applied to Q or K at all.

By comparing these three, we determine whether rank dynamics are driven by the normalization itself (gradient stabilization) or by the parameter $\gamma$ selectively modulating dimensions.

**Key result:** The learned $\gamma$ is the dominant driver of rank preservation. At 50M tokens, Learned $\gamma$ maintains a mean PR of **65.7**, while Frozen $\gamma=1$ collapses to **59.6** — lower than even the No QK-Norm baseline (**63.3**). Normalization without learnable $\gamma$ actively hinders rank recovery.

### Architecture & Training Config

| Parameter | Value |
|:---|:---|
| **Model Size** | ~1.58B Parameters |
| $d_\text{model}$ | 2048 |
| Layers | 32 |
| Attention Heads | 16 ($d_k = 128$) |
| KV Heads | 8 (GQA) |
| $d_\text{ff}$ | 8192 |
| **Training Budget** | **~49M Tokens** per condition |
| **Hardware** | NVIDIA H100 80GB |
| Batch Size | 8 |
| Grad Accumulation | 4 |
| Effective Batch | $8 \times 4 \times 2048 = 65{,}536$ tokens/step |
| Optimizer | Muon + AdamW |
| Dataset | FineWeb/Cosmopedia Mix (1B subset) |

---

## 2. Hypothesis & Methodology

We track the **Participation Ratio (PR)** of the key representations $K$ at regular intervals (every 120 steps, ≈ 2M tokens), yielding 26 measurements per condition.

### The Causal Triangulation: How Three Conditions Let Us Isolate the Cause

The core question is: **What makes QK-Norm different from no normalization — is it the normalization step itself, or the learned γ weights?**

QK-Norm does two things at once: (1) it normalizes the vector, and (2) it applies learnable γ scaling. By adding a third condition (Frozen γ=1), we surgically remove the γ learning while keeping the normalization. Then we compare the PR (our measure of dimensional diversity) across all three:

- **Scenario A — γ is the driver:** If the Learned γ model shows different PR than both Frozen and NoQK, but Frozen and NoQK look similar to each other, then it must be the γ learning that matters — because removing γ learning (Frozen) makes the model behave like having no norm at all.

- **Scenario B — Normalization is the driver:** If Learned γ and Frozen show similar PR (both have normalization), but NoQK is different, then the normalization step itself is what matters — γ learning is irrelevant.

- **Scenario C — Both contribute:** If all three show different PR values, then both the normalization and γ learning play a role.

**Our actual result matches Scenario A** at the collapse floor, and a surprising variant afterward: Frozen γ ends up *worse* than NoQK, meaning normalization without γ learning is actively harmful.

## 3. Results

### 3.1 Summary

| Metric | Learned γ | Frozen γ=1 | No QK-Norm |
|:---|:---:|:---:|:---:|
| **Initial Mean PR** | 86.85 | 86.85 | 86.42 |
| **Minimum Mean PR** | 55.17 (7.9M) | 51.84 (9.8M) | 51.28 (7.9M) |
| **Final Mean PR (49M)** | **65.70** | 59.55 | 63.25 |
| **Final Train Loss** | **3.618** | **3.614** | 3.683 |
| **Final Val Loss** | **3.440** | **3.439** | 3.505 |

### 3.2 The PR Trajectory: Collapse → Recovery → Plateau

![Key Rank Collapse — Does γ or Normalization Drive It?](./images/2_key_pr_causal.png)

All three conditions follow a **U-shaped trajectory**:

1. **Rapid Collapse (0–8M):** PR drops from ~87 to ~51–55 as random-init structure is destroyed.
2. **Recovery (8M–25M):** PR climbs back to ~60–66 as meaningful representations form.
3. **Plateau (25M–49M):** PR stabilizes with no significant further change.

**The causal signal is clearest during recovery:**
- Learned γ recovers to 65.7 (highest)
- No QK-Norm recovers to 63.3
- Frozen γ=1 recovers only to 59.6 (lowest)

**Verdict:** Frozen γ ≈ NoQK at the collapse floor → normalization alone doesn't prevent collapse. But Frozen γ < NoQK after recovery → normalization without learnable γ *hurts* rank recovery. The learned $\gamma$ is the mechanism.

### 3.3 Loss vs. Rank Dissociation

![Training & Validation Loss](./images/1_loss.png)

Normalization helps **loss** identically regardless of γ learning (Learned γ ≈ Frozen γ ≈ 3.62, both lower than NoQK = 3.68). But normalization helps **rank** only when γ is learnable. The mechanism that optimizes loss (gradient stabilization) is different from the mechanism that preserves rank (γ-mediated dimension selection).

## 4. Mathematical Background

### 4.1 QK-Normalization

In standard attention, a token representation $\mathbf{x} \in \mathbb{R}^{d_\text{model}}$ (a 2048-dimensional vector representing one token) is projected into queries and keys:

$$Q = XW_Q, \quad K = XW_K$$

where:
- $X \in \mathbb{R}^{n \times d_\text{model}}$ is the input matrix ($n$ tokens, each a 2048-dim vector)
- $W_Q \in \mathbb{R}^{d_\text{model} \times d_k}$ is the query weight matrix (projects from 2048 → 128 dimensions per head)
- $W_K \in \mathbb{R}^{d_\text{model} \times d_k}$ is the key weight matrix (same shape)
- $d_k = 128$ is the per-head dimension

With QK-Norm (as in our Gemma-style model), RMSNorm is applied to **both Q and K independently**, before rotary position encoding (RoPE):

$$Q = \text{RoPE}\!\left(\text{RMSNorm}(XW_Q)\right), \quad K = \text{RoPE}\!\left(\text{RMSNorm}(XW_K)\right)$$

The RMSNorm operation for a single vector $\mathbf{x} \in \mathbb{R}^{d_k}$ works as follows:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d_k}\sum_{i=1}^{d_k} x_i^2}$$

$$\text{RMSNorm}(\mathbf{x})_j = \frac{\gamma_j \cdot x_j}{\text{RMS}(\mathbf{x})}$$

where:
- $\text{RMS}(\mathbf{x})$ is a single scalar — the root-mean-square of all 128 values in the vector
- $x_j$ is the value at dimension $j$ (before normalization)
- $\gamma_j$ is a learnable scalar weight for dimension $j$, initialized to 1.0
- The output at dimension $j$ is the original value divided by the RMS, then scaled by $\gamma_j$

**Scope of γ:** Each RMSNorm layer has its **own independent** $\gamma \in \mathbb{R}^{128}$ vector. Since QK-Norm applies RMSNorm to both Q and K at every layer, there are $32 \text{ layers} \times 2 \text{ (Q and K)} = 64$ separate $\gamma$ vectors in the model, totaling $64 \times 128 = 8{,}192$ additional learnable parameters.

**What γ controls:** At initialization ($\gamma = \mathbf{1}$), the norm simply rescales the vector to have unit RMS. As training progresses, individual $\gamma_j$ values diverge — some grow above 1 (amplifying that dimension) and some shrink toward 0 (suppressing it). This gives the model a direct, per-dimension knob to control which dimensions of the key/query space are important.

**What RoPE does:** Rotary Position Encoding ($\text{RoPE}$) applies a position-dependent rotation to pairs of dimensions in the Q and K vectors. It encodes positional information without adding parameters. It is applied **after** RMSNorm, so $\gamma$ acts on the pre-RoPE representation.

### 4.2 Why This Matters for Rank

The **Participation Ratio (PR)** measures how many dimensions of the key space are "actively used" — i.e., how evenly the information is spread across the 128 dimensions of each attention head.

To compute it, we collect the key vectors $K \in \mathbb{R}^{n \times 128}$ from many tokens and compute the Singular Value Decomposition (SVD), yielding singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{128}$. These singular values tell us how much variance each dimension captures:

$$\text{PR}(K) = \frac{\left(\sum_{i=1}^{d_k} \sigma_i\right)^2}{\sum_{i=1}^{d_k} \sigma_i^2}$$

**Intuition:**
- If all 128 singular values are equal ($\sigma_i = c$), then $\text{PR} = 128$ — all dimensions contribute equally (full rank)
- If only one singular value is nonzero ($\sigma_1 = c$, rest = 0), then $\text{PR} = 1$ — total collapse to a single dimension
- A PR of 60 (roughly what we observe at convergence) means the information is spread across roughly 60 effective dimensions

**How each condition affects rank:**
- **Without QK-Norm**: The singular values of $K$ are determined solely by $W_K$ and the input data. The model can only control rank through the weight matrix.
- **With QK-Norm (Learned γ)**: The explicit $\gamma$ parameter can independently amplify or suppress each dimension — giving the model a direct, low-resistance path to shape the spectrum.
- **With QK-Norm (Frozen γ=1)**: Normalization constrains key norms but without per-dimension flexibility, creating a geometric constraint that limits spectral diversity.

## 5. γ Dynamics

The γ coefficient of variation (CV) reveals how the model uses the learnable parameter:

- **High γ CV layers** (L0: 0.168, L15: 0.155, L27: 0.172): Aggressive dimension differentiation, correlated with higher PR recovery.
- **Low γ CV layers** (L5–L10: 0.07–0.08): Minimal differentiation, correlated with the lowest PR values.
- **Frozen γ=1**: CV = 0.0 at all layers (confirming ablation correctness).

γ acts as a **dimension-selective amplifier** that preserves spectral diversity, rather than simply suppressing "dead" dimensions.

## 6. Limitations

1. **Single Seed**: We run one seed per condition. The effect size (2–6 PR units) is larger than the 500K pilot (0.75 units), increasing confidence, but variance estimates require multiple seeds.
2. **Early Training**: 50M tokens is ~0.3% of Chinchilla-optimal for a 1.5B model. We observe early-phase dynamics, not convergence behavior.
3. **Key-Only Analysis**: We measure the PR of Keys ($K$). The effective rank of the full attention matrix ($QK^T$) also depends on Queries ($Q$), which we do not probe.
4. **Muon Optimizer**: Muon's orthogonal updates may interact with spectral dynamics in ways specific to this optimizer.

## 7. Conclusions

1. **γ drives rank, normalization drives loss.** These are separable mechanisms. QK-Norm's benefit comes from two independent contributions.
2. **PR follows a U-shaped trajectory** (collapse → recovery → plateau), a structural feature of early training.
3. **Normalization without γ is counterproductive for rank.** Frozen γ=1 produces the worst final PR — lower than even the no-normalization baseline.
4. **The depth-PR gradient inverts** between early (500K) and extended (50M) training.

### Next Steps

- **Track individual γ values** to determine which dimensions are amplified vs. suppressed
- **Extend to 250M+ tokens** for longer-horizon dynamics
- **Measure query PR** alongside key PR
- **Multiple seeds** (≥3) for variance estimation
- **Pure AdamW baseline** to isolate Muon's role
