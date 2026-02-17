# Analysis of Attention Key Evolution

---

## 1. The Energy of Attention (Key Norms)

### 🧠 Prerequisite: The L2 Norm
Before looking at the data, we must understand the baseline energy of these vectors.
*   **Definition:** The L2 Norm ($\|x\|_2$) measures the total magnitude or "length" of a vector.
*   **The "Magic Number" 8.0:** In this model (`d_model=512`, `heads=8`), the head dimension is $d_k=64$. Since `RMSNorm` normalizes the vector such that the **mean of its squared components** ($\frac{1}{d_k}\sum x_i^2$) equals 1, the total squared sum is $d_k \times 1 = 64$. Therefore, the expected L2 Norm is $\sqrt{64} = 8.0$.
#### ❓ Theory: Why "fight" the norm?
The length of the Key vector directly controls the **sharpness** (or temperature) of the attention mechanism. Since Attention is $Softmax(Q \cdot K)$, the magnitude of $K$ scales the logits.
*   **Dampening (Norm < 8.0 = High Temp):** The model shrinks vectors to make dot products smaller.
    *   *Effect:* The Softmax input becomes flatter. The model attends to **everything equally** (High Entropy). It is "averaging" the context.
*   **Amplification (Norm > 8.0 = Low Temp):** The model grows vectors to make specific dot products massive.
    *   *Effect:* The Softmax output approximates an `argmax`. The model attends to **one specific token** (Low Entropy). It is "selecting" information.

#### 🧪 How to Prove It (Simple Experiment)
If this theory is true, Norm and Attention Entropy must be inversely correlated.
1.  **Measure:** For a batch of data, calculate the **Shannon Entropy** of the attention weights (post-softmax) for each layer.
    *   Formula: $H(p) = - \sum p_i \log(p_i)$
2.  **Verify:**
    *   **Layer 0 (Norm ~5.3):** Should have **High Entropy** (diffuse/blurry attention).
    *   **Layer 21 (Norm ~9.8):** Should have **Low Entropy** (sharp/spiky attention).

### 📊 Analysis
![Key Norm Evolution](key_norm_evolution.png)
*Figure 1: Evolution of the average Key Norm over 1000 steps.*

The plot reveals three distinct behaviors based on layer depth:
*   **Early Layers (Blue):** The norm steadily decays from the initialized 8.0 down to ~5.3. This suggests "feature dampening" where the model reduces the influence of raw embeddings.
*   **Middle (Green) & Final (Red) Layers:** After an initial dip, these layers aggressively pump energy into the keys, rising to ~9.5–9.8. This amplification helps these features survive the softmax bottleneck in attention, making them "sharper" and more dominant.

---

## 2. The Birth of Attention Sinks (Outliers)

### 🧠 Prerequisite: Max Absolute Value ($max\_abs$)
While the Norm measures *total* energy, we also need to know how that energy is distributed.
*   **Definition:** This tracks the magnitude of the *single largest component* in the vector.
*   **The "Sink" Indicator:** If a vector has a norm of 8.0 but a $max\_abs$ of 7.5, it means ~88% of the vector's power ($7.5^2 / 8.0^2$) is concentrated in just **one dimension**.
*   **Implication:** This creates a "hard switch." If a Query matches this single dimension, it will trigger a massive attention score, effectively creating an **Attention Sink**.

### 📊 Analysis
![Outlier Magnitude](outlier_magnitude.png)
*Figure 2: The maximum absolute value of any single component in the key vectors.*

This plot provides the strongest evidence for Attention Sinks:
*   **The Spike:** In Middle (Green) and Final (Red) layers, the $max\_abs$ value rockets from ~4.0 to nearly 9.0 very quickly (steps 100-200).
*   **Comparison to Norm:** Note that while the *Norm* (Fig 1) is ~9.5, the *Max Value* (Fig 2) is ~9.0.
*   **Conclusion:** Almost **all** of the vector's energy is concentrated in a single dimension. The vectors are not distributed; they are axis-aligned spikes.

---

## 3. Semantic Collapse (Latent Redundancy)

### 🧠 Prerequisite: Cosine Similarity
How different are the keys from each other?
*   **Definition:** We measure the average **Cosine Similarity** between random pairs of keys within the same layer.
*   **Low Similarity (~0.0):** Vectors are orthogonal (diverse). The layer is preserving distinct feature information.
*   **High Similarity (>0.5):** Vectors are pointing in the same direction. The layer is collapsing representations into a shared semantic space.

### 📊 Analysis
![Latent Redundancy](latent_redundancy.png)
*Figure 3: Average cosine similarity between random key pairs.*

*   **Early Layers (Blue):** Similarity stays near 0.05. This confirms **Feature Diversification**. The model keeps these representations distinct to capture fine-grained token differences.
*   **Deep Layers (Green/Red):** We see a rapid convergence to >0.5 similarity.
*   **Mechanism:** This indicates **Semantic Convergence**. By the deeper layers, the model is no longer tracking specific token identities but rather their broad semantic roles. The keys become "redundant" because they all point toward the same prediction-relevant concepts.

---

## 4. Structural Hierarchy

### 📊 Analysis
![Key Magnitude Profile](key_magnitude_profile.png)
*Figure 4: The final average L2 norm for each layer at step 1000.*

This bar chart captures the final structural state of the model:
*   **The "Checkmark" Pattern:**
    *   **Layers 0-3:** Low energy (~3.0 - 6.0). The "Encoder" phase of processing raw signals.
    *   **Layers 4-21:** Linearly increasing magnitude.
    *   **Implication:** The model constructs a hierarchy of importance. Deeper layers are granted higher magnitude (and thus lower temperature/sharper attention) to make definitive predictions for the next token.

---

## 🚀 Implications for Future Research

1.  **Quantization Sensitivity:** The high-magnitude outliers ($max\_abs > 7.5$) shown in Figure 2 require high precision. Naive quantization would clip these spikes, destroying the attention mechanism.
2.  **Regularization:** The divergence of norms in Figure 1 suggests that regularizing $max\_abs$ could force the model to distribute information more evenly, potentially preventing sink formation.
3.  **Pruning Opportunities:** The high redundancy observed in Figure 3 suggests that deep layers (19-21) might be rank-deficient and candidates for pruning without significant loss of performance.
