# Curvature-Aware Muon (C-Muon): Making AI Training More Stable and Reliable
**Authors: Vuk Rosić and Gemini**

## Abstract
Training Large Language Models (LLMs) is like driving a heavy vehicle across a steep mountain range—a mathematical "loss landscape" where our goal is to reach the lowest valley (the point of minimum error). A recent breakthrough optimizer called **Muon** speeds this up by ensuring its update steps are **orthogonal**, meaning each move is completely independent of the last. This independence prevents the model from wasting energy in redundant, zigzagging motions. However, Muon can become unstable in "jagged" regions of the landscape, known as areas of **high curvature**. In these rough spots, a standard Muon step might be too aggressive, overshooting the path and causing the training process to fail or "diverge."

We propose **Curvature-Aware Muon (C-Muon)** to address this instability. Inspired by **Ricci Flow**—a geometric process used in advanced mathematics to "iron out" wrinkles and smooth out irregular surfaces—C-Muon acts as a local terrain sensor. it monitors how quickly the landscape is changing and applies a smoothing dampener to the updates *before* they are finalized. This ensures that the model slows down or smooths its path when the terrain gets too jagged, leading to more reliable training without adding significant complexity. We provide a practical implementation recipe that allows researchers to easily integrate curvature-awareness into their existing AI training pipelines.

## 1. Introduction
### 1.1 Motivation
In the race to build more powerful AI, the "engine" of the process is the optimizer. Traditional optimizers like AdamW are reliable but can be slow because they often repeat information across different parts of the model. **Muon** has recently demonstrated that we can train models much faster by forcing updates to be "orthogonal" (independent). 

However, neural network landscapes are notoriously "bumpy." In regions of extreme curvature—where the slope of the hill changes very suddenly—taking a large independent step is risky. It's like trying to take a wide, confident stride on a surface full of narrow cracks; you are likely to trip. These "trips" in AI training appear as massive spikes in the loss graph, often forced researchers to restart expensive training runs from scratch.

### 1.2 Objective
This paper introduces **C-Muon**. Our objective is to give the Muon optimizer a "sense of touch" for the terrain it is navigating. By incorporating a discrete approximation of **Ricci Flow**, we allow the optimizer to proactively smooth out its own gradient updates in response to detected "jaggedness," thereby preventing divergence while maintaining the speed of orthogonal optimization.

## 2. Background and Related Work
### 2.1 Muon and Orthogonalization
Muon utilizes a technique called the **Newton-Schulz iteration** to perform orthogonalization. Given a gradient matrix $G$, Muon finds a matrix $U$ that carries the same "direction" as $G$ but satisfies $U^T U = I$ (where $I$ is the identity matrix, representing perfectly independent dimensions). The core iteration is:
$$X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k)$$
This process effectively "decorrelates" the features the model is learning, allowing it to learn many things in parallel rather than one after another.

### 2.2 Ricci Flow: The Architecture of Smoothing
In differential geometry, a **manifold** is a mathematical name for a surface (like the skin of an orange or the shape of a mountain range). **Ricci Flow** is a famous equation that describes how to smooth a manifold over time:
$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij}$$
Here, $g$ is the "metric" (the ruler we use to measure distance on the surface) and $R$ is the "Ricci curvature" (the measure of how much the surface is curved or wrinkled). Essentially, Ricci flow says: "Shrink the bumps and expand the valleys until the surface is smooth."

## 3. Methodology: The C-Muon Framework
### 3.1 Defining the Metric and Curvature
To bring Ricci Flow into the world of AI, we must first define our "surface." We treat each parameter of the model as a point on a manifold. We define a local **Metric $\hat{g}$** based on the recent "steepness" of the loss at that point (using the moving average of squared gradients):
$$\hat{g}_{i, t} = \beta_2 \hat{g}_{i, t-1} + (1 - \beta_2) G_{i, t}^2$$
Think of $\hat{g}_i$ as a "memory" of how steep the terrain has been at parameter $i$.

The **Scalar Curvature $R$** measures how much this steepness *changes* as we move between neighboring parameters. In a computer, we can't use continuous calculus, so we use a **discrete second difference**, which is the mathematical way of measuring "acceleration" or "change in slope":
$$R(\hat{g}) \approx \sum_{i} \left| \hat{g}_{i+1} - 2\hat{g}_i + \hat{g}_{i-1} \right|$$
If $\hat{g}$ is the same for all neighbors, the landscape is "flat" and $R=0$. If one parameter is much steeper than its neighbors, $R$ will be high, signaling a "jagged" region.

We compute the final curvature weight $\kappa$:
$$\kappa = \text{Softplus}(R(\hat{g}))$$
The `Softplus` function ($\ln(1 + e^x)$) ensures $\kappa$ is always a positive number that smoothly approaches zero in calm regions.

### 3.2 The Smoothing Step (Internal Ricci Flow)
Once we know the curvature, we apply a "mini" Ricci flow to our gradients before they reach the Muon orthogonalization step. We damp the original gradient $G$ by an exponential factor:
$$G_{smoothed} = G \cdot \exp(-\alpha \cdot \kappa)$$
The hyperparameter $\alpha$ (Alpha) acts like a "sensitivity" setting. If curvature is high, the gradient is significantly softened, preventing the optimizer from making an aggressive move into a jagged cliff.

### 3.3 The C-Muon Algorithm
1.  **Step 1: Metric Tracking.** Maintain a moving average $\hat{g}$ of squared gradients.
2.  **Step 2: Curvature Sensing.** Compute the absolute second-order variation of $\hat{g}$ across the parameter indices.
3.  **Step 3: Smoothing.** Scale the current gradient by $\exp(-\alpha \cdot \text{Softplus}(\text{Sum}(\text{variations})))$.
4.  **Step 4: Orthogonalization.** Pass this smoothed gradient into the standard Muon Newton-Schulz (or Polar Express) logic.
5.  **Step 5: Update.** Apply the resulting orthogonal update to the model weights.

## 4. Mathematical Derivation (For Undergraduates)
### 4.1 Why Orthogonalize?
Imagine you are playing a team sport. If everyone on the team runs toward the ball at the same time, they collide and leave the rest of the field open. If you force the players to spread out and move in "orthogonal" (independent) directions, they cover the entire field much more efficiently. Muon does this for the "neurons" in an AI model.

### 4.2 The Intuition of Curvature
If you are walking on a smooth, flat floor, several large steps are fine. If you are walking on a pile of loose rocks, those same large steps will make you slip. Curvature $R$ is our way of telling the model: "The floor is turning into rocks, take smaller, more careful steps."

### 4.3 Why the Second Difference?
- **First Difference ($g_{i+1} - g_i$):** Tells us the slope (how steep it is).
- **Second Difference ($g_{i+1} - 2g_i + g_{i-1}$):** Tells us how the slope is *changing*. In physics, this is like acceleration. If the slope changes from "up" to "down" instantly, that's high curvature, and it means there's a sharp peak or valley that could cause a divergence spike.

## 5. Proposed Experiments
### 5.1 Training Stability Tests
We will test C-Muon on the `train_llm.py` setup using a GPT-2 style architecture (12 layers, 768 hidden dimension) trained on the FineWeb-Edu dataset.
- **8M Token Run:** A quick "smoke test" to see if C-Muon can match or exceed standard Muon's speed.
- **20M Token Run:** A longer run to observe stability. We will intentionally increase the learning rate to find the "breaking point" of both optimizers. We hypothesize that C-Muon will remain stable at learning rates where standard Muon fails.

### 5.2 Metrics for "Success"
1.  **Final Validation Loss:** Does smoothing help the model reach a better final state?
2.  **Loss Spike Frequency:** We will count the number of "mini-divergences" (where the loss jumps significantly in one step).
3.  **Wall-clock Efficiency:** Does the curvature calculation slow down the training? (Expected overhead: $<0.5\%$ per step).

## 6. Discussion and Future Work
C-Muon provides a simple yet mathematically grounded bridge between raw optimization and the geometric reality of loss landscapes. By using a discrete version of Ricci Flow, we transform the optimizer from a blind "independent stepper" into an informed "terrain navigator."

**Next Steps:**
- **Auto-Alpha:** Can we make the sensitivity parameter $\alpha$ learnable?
- **Layer-wise Curvature:** Currently, we compute a single $\kappa$ for the whole tensor. It may be more effective to compute it per-neuron or per-head.
- **Deeper Geometry:** Exploring more advanced curvature tensors (like the Riemann tensor) for even more precise terrain sensing.

