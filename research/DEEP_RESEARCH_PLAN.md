# Research Plan: Strategic Divergence of Attention Heads
**Objective:** Investigate whether Transformer layers hide a "Civil War" of specialization where individual heads within a single layer split into radically different structural roles (Diffuse Context vs. Sharp Sinks).

---

## 🔬 Phase 1: Tracking Internal Head Variance
Instead of averaging Key Norms per layer, we must track the **inter-head distribution**.

### 🛠 Modification:
*   **Metric:** Track `L2_Norm` and `Max_Abs` for *every* individual head $(h_1, h_2 ... h_8)$ independently.
*   **GQA Warning:** If using Grouped Query Attention, track the $KV$ heads specifically, as they are the ones controlled by the `k_norm` gain.
*   **Key Question:** Does a "Middle Layer" have an average norm of 9.0 because all heads are 9.0, or because 4 heads are "Sinks" (Norm: 15.0) and 4 heads are "Gatherers" (Norm: 3.0)?

---

## 🧠 Phase 2: The Entropy-Magnitude Correlation
Prove that high Key Norm = Physical Sharpness.

### 🧪 Experiment:
1.  **Calculate Shannon Entropy** $H$ of the attention pattern for every head: $H(head) = -\sum p_i \log p_i$.
2.  **Plot Scatter Map:** X-axis = `Key Norm`, Y-axis = `Attention Entropy`.
3.  **Hypothesis:** We will see a multi-modal distribution.
    *   **Cluster A (The Gatherers):** Low Norm, High Entropy. These heads "blur" the past to understand broad context.
    *   **Cluster B (The Sinks):** High Norm, Low Entropy (near 0). These heads are "fixed switches" pointing at `Token 0` or punctuation.

---

## 📐 Phase 3: Dimensional Collapse (Effective Rank)
Investigate the **Axis-Alignment** of outliers.

### 🔍 Analysis:
*   **Measurement:** Compute the **Effective Rank** (via Participation Ratio or Singular Value Decomposition) of the Key matrix $K$ within each head.
*   **Thresholding:** When $max\_abs$ approaches the $L2\_Norm$, the head is effectively "one-hot."
*   **Key Question:** Is the magnitude growth uniform across the head's vector, or is the model "investing" all energy into a single "privileged dimension" to bypass the softmax bottleneck?

---

## 🚀 Phase 4: The Dynamic Range Test (The "Volume Knob")
Determine if the model controls focus through $K$ (Keys) or $Q$ (Queries).

### 🧪 Experiment:
*   Track the ratio of `||Q|| / ||K||`. 
*   If `||Q||` stays constant at 8.0 but `||K||` explodes, the model is using the Key gain as a systemic "sensitivity knob."
*   If both grow, it is a coordinated "Focus Peak."

---

## 📈 Success Metric
If this plan works, you should find that **Layer Depth is NOT the only variable**. Instead, you will see a **Functional Taxonomy of Heads**:
1.  **Static Sinks** (High Norm, Axis-Aligned, Constant across tokens)
2.  **Semantic Retrieval** (High Norm, Query-Dependent focus)
3.  **Syntactic Gatherers** (Low Norm, Broad Entropy, consistent Across sequences)

---

### 💻 Execution Snippet (for your implementation)
```python
# Insert this logic into your k_norm hook
def hook(module, input, output):
    # output: [B, T, H, D]
    with torch.no_grad():
        # 1. Per-head norms: [H]
        head_norms = torch.norm(output, dim=-1).mean(dim=(0, 1)) 
        
        # 2. Per-head max outliers: [H]
        head_max = output.abs().max(dim=-1)[0].mean(dim=(0, 1))
        
        # 3. Participation Ratio (Effective Rank) per head
        # PR = (sum(x_i^2))^2 / sum(x_i^4)
        sq = output.pow(2).sum(dim=-1)
        quad = output.pow(4).sum(dim=-1)
        pr = (sq.pow(2) / quad).mean(dim=(0, 1)) # [H]
        
        # Store these arrays to see how heads diverge over time
```
