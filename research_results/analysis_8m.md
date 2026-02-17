# What happens with attention keys as LLM trains?

![Key Analysis Report](key_analysis_report.png)
*Figure 1: (A) Smoothed Norm evolution showing stage-based decay. (B) Outlier spikes highlighting sink formation. (C) Cleaned Latent Redundancy showing the convergence in deep layers.*

---

## 📊 Overview
This report details the tracking of key vector statistics during an 8-million-token training run of a transformer model. By monitoring norms, outlier magnitudes, and latent redundancy, we've identified a clear architectural specialization that the model develops to manage attention flow.

---

## 🔍 Fundamental Metrics

### 1. Vector Norm (L2)
The **L2 Norm** represents the total "energy" or magnitude of a key vector.

#### 💡 Why is the Norm exactly 8.0?
The value is not arbitrary; it is a direct mathematical consequence of the **Head Dimension**.
- In this model, `d_model = 512` and `n_heads = 8`.
- This results in a **Head Dimension ($d_k$)** of $512 / 8 = 64$.
- The model uses `RMSNorm` on the Keys. By definition, Root Mean Square normalization sets the average squared value of components to 1.
- Mathematically: $\|x\|_2 = \sqrt{d_k} = \sqrt{64} = \mathbf{8.0}$.
- **Insight:** This constant magnitude ensures that attention scores aren't sensitive to initial embedding scale, but it forces the model to create "Attention Sinks" by concentrating that 8.0 energy into a single "peaky" dimension.

### 2. Max Absolute Value ($max\_abs$)
This tracks the single largest component within a vector. It is our primary signal for **"Peakiness."**
- **The Sink Signal:** If a vector has a norm of 8.0 but one component is 7.5, it means ~88% of the vector's power ($7.5^2 / 8.0^2 = 56.25 / 64$) is concentrated in just **one dimension**. This creates a massive "alignment spike" whenever a Query's dimension matches it.

### 3. Latent Redundancy (Cosine Similarity)
This tracks how similar random pairs of keys are within a layer. 

---

## 📈 Key Research Findings

### 🏙️ 1. Early Layers: Feature Diversification
In Layers 0-4, we observe **high dimensional diversity**.
- **Observation:** Cosine similarity remains near 0.1.
- **Dynamics:** The model is projecting raw token embeddings into a broad feature space. Keys are kept unique to ensure the attention mechanism can distinguish between subtle contextual differences.

### 🕳️ 2. Middle Layers: The Birth of Attention Sinks
In Layers 10-18, a striking **directional specialization** occurs.
- **The "Sink" Spikes:** While the total norm is fixed, the `max_abs` value spikes to **>7.5**.
- **Strategic Concentration:** This represents the emergence of **Attention Sinks**. The model is essentially turning specific tokens into "black holes" for attention. 

### 🧘 3. Final Layers: Semantic Convergence
As we approach the final layers (19-21), indices move toward **Systemic Redundancy**.
- **Observation:** Average cosine similarity rises consistently.
- **Mechanism:** The representation becomes semantically aligned as the model prepares for next-token prediction. Features converge from specific token positions into broad conceptual clusters.

---

## 🚀 Implications for Future Research

1.  **Quantization Sensitivity:** The high-magnitude outliers ($max\_abs > 7.5$) require high precision to maintain the Attention Sink mechanism.
2.  **Regularization:** Penalizing $max\_abs$ during training could force the model to be more "attentive" to local context rather than relying on sinks.
3.  **Efficiency:** Pruning deep layers might be easier due to the observed high redundancy (Convergence).
