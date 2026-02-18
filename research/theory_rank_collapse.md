# Theory of Dimensional Collapse in Normalized Attention

## 1. The Geometry of the Softmax Bottleneck

Transformer attention is defined by the Softmax operation:
$$A(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

To minimize the cross-entropy loss, the model must often attend to very specific tokens (e.g., the preceding word in a common phrase). This requires the attention distribution to be "sharp" (low entropy).

Mathematically, to make a distribution spiky, the dot products $q_i \cdot k_j$ must have a high dynamic range. Specifically, one dot product must be much larger than the others.

## 2. Why Normalization Accelerates Collapse

### The "Energy Budget" Argument
When we use **QK-Normalization** (RMSNorm or LayerNorm on $Q$ and $K$), we constrain the L2-norm of the vectors:
$$\|q\|_2 \approx \sqrt{d_k}, \quad \|k\|_2 \approx \sqrt{d_k}$$

In an un-normalized transformer, the model can make a dot product large by simply growing the magnitudes of $W_q$ and $W_k$ (global scaling). However, in a normalized transformer, the "volume knob" is broken. You cannot grow the vector length beyond the fixed quota.

The only remaining way to increase a dot product $q \cdot k = \|q\|\|k\|\cos(\theta)$ is to **minimize the angle $\theta$**.

### The Axis-Alignment Shortcut
In high-dimensional space ($d_k = 64, 128$), it is much easier for an optimizer to align $Q$ and $K$ along a single, shared coordinate axis than to maintain a complex, high-rank representation.

If a head learns to focus on a specific feature (like "is this token a comma?"), it projects all relevant information onto **one dimension**. Because the total energy is fixed (by the Norm), if the energy in Dimension $X$ goes up, the energy in all other dimensions **must** go down to compensate.

This creates a **Winner-Take-All** effect:
1. The model finds a useful "privileged dimension."
2. The optimizer pushes energy into that dimension to maximize attention sharpness.
3. Normalization forces other dimensions to zero to maintain the norm.
4. The weight matrix effectively collapses into a rank-1 or low-rank projection.

## 3. The Optimizer's Role (AdamW vs Muon)

### AdamW (Coordinate-wise Collapse)
AdamW tracks the moving average of gradients for each weight individually. If a specific dimension in the Key projection starts providing a clear signal, AdamW will aggressively step in that specific coordinate direction. This "coordinate-wise" update style perfectly complements the axis-aligned collapse of normalized layers.

### Muon (Spectral Resistance)
Muon (Newton-like Orthogonalization) applies updates that keep the weight matrix $W$ "near-orthogonal" ($W^T W \approx I$). 
Muon explicitly fights rank collapse in the weights themselves. Even if the training signal wants to compress the representation, Muon's spectral pressure tries to keep all dimensions alive. 

**This is why Muon without QK-Norm is the "Gold Standard" for internal representation health.**

## 4. The Stability Paradox

The paradox observed in our experiments—that **QK-Norm has better loss but worse rank**—suggests that LLMs are "lazy" by default. 

- **QK-Norm** provides a shortcut to low loss by providing a sharp, low-rank mechanism for retrieving recent tokens and attention sinks.
- **No QK-Norm** forces the model to build more robust, high-rank representations, which might be slower to learn initially but produce a "healthier" internal state that likely scales better to long-context and complex reasoning.

## 5. The Participation Ratio (PR) Metric

To measure the effective rank, we use the **Participation Ratio**, which is more robust than a hard threshold on singular values.

For a matrix with singular values $\sigma_1, \sigma_2, \dots, \sigma_{d_k}$:

$$PR = \frac{(\sum_{i=1}^{d_k} \sigma_i)^2}{\sum_{i=1}^{d_k} \sigma_i^2}$$

### Intuition:
- **Maximum Diversity**: If the representation is perfectly distributed and all singular values are equal ($\sigma_i = \sigma$), the formula reduces to $\frac{(d_k \cdot \sigma)^2}{d_k \cdot \sigma^2} = d_k$.
- **Complete Collapse**: If the representation collapses onto a single dimension ($\sigma_1 = 1$, others $= 0$), the formula reduces to $\frac{1^2}{1^2} = 1$.

The denominator $\sum \sigma_i^2$ (the Frobenius norm squared) is dominated by the largest singular values. If a few "outliers" become much larger than the rest, the denominator grows disproportionately, causing the ratio to plummet towards 1. 

This metric captures the **effective dimensionality** being utilized by the model's "internal sensors."

**Final Recommendation**: 
- If using AdamW: Use QK-Norm for stability but watch for rank collapse.
- If using Muon: Consider disabling QK-Norm to preserve internal diversity.
