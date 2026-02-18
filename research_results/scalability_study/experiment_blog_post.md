# 🧪 Experiment: Does QK-Norm Cause Brain Damage at Scale?

**Status: Training in Progress (300M Token Run)**

We are running a rigorous 300M token ablation study to confirm a counter-intuitive hypothesis: **QK-Normalization, while stabilizing early training, acts as a "ceiling" that destroys representational capacity in the long run.**

## 1. The "Short-Run Deception" Hypothesis

In short pilot runs (1M tokens), QK-Normalization—applying RMSNorm to Queries and Keys before attention—appears to be a clear winner. It stabilizes gradients, prevents attention logits from exploding, and generally makes training safer.

However, our preliminary data suggests this is a trap.

**The Hypothesis:**
1.  **The Constraint Trap:** By forcing keys onto a unit hypersphere, QK-Norm limits the geometric space available for representations.
2.  **Optimizer Behavior:** Coordinate-wise optimizers like **AdamW** find the path of least resistance on this sphere, which often involves collapsing multiple attention heads into the same few axis-aligned directions.
3.  **The Reversal:** While QK-Norm wins at 1M tokens, we predict it will cause massive **Dimensional Collapse** (measured by Participation Ratio) at 300M tokens.
4.  **The Muon Alternative:** We hypothesize that **Muon** (an orthogonal optimizer) can maintain stability *without* normalization by naturally pushing representations apart, preserving the full dimensionality of the model.

## 2. Experimental Setup

To prove this, we are training **8 models** (4 configurations x 2 seeds) for **300 Million Tokens** each.

### The Model Architecture (MinimalLLM)
We are using a dense 88M parameter transformer designed to mimic modern insights (RoPE, RMSNorm, SwiGLU).

*   **Parameters:** ~88M
*   **Layers:** 22
*   **Heads:** 8 (Head Dim = 64)
*   **Context Length:** 2048
*   **Dataset:** FineWeb-Edu + Cosmopedia Mix (High quality synthetic/web mix)
*   **Precision:** bfloat16 (AMP)

### The 4 Configurations
We are testing the interaction between **Optimizer** and **normalization**:

1.  **Baseline (AdamW + QK-Norm):** The industry standard safety config.
2.  **AdamW (No QK-Norm):** Can AdamW survive without the guardrails?
3.  **Muon (No QK-Norm):** **(Our Hero)** Can orthogonal optimization replace explicit normalization?
4.  **Muon + QK-Norm:** Does normalization hurt Muon's ability to learn high-rank features?

## 3. Rigorous Methodology

We are controlling for every variable to ensure the results are widely applicable:

*   **Fixed Validation Set:** A frozen 5M token slice of the dataset, never seen during training, used to compare loss fairly across all runs.
*   **Independent LR Tuning:** We are sweeping learning rates for *every single configuration* independently (20 short runs) to ensure we compare the *best* version of AdamW against the *best* version of Muon.
*   **Metrics:**
    *   **Validation Loss:** Predicting the next token.
    *   **Participation Ratio (PR):** A spectral metric measuring the effective dimensionality of the attention Key ($K$) matrices. $PR=64$ = Full Rank. $PR=1$ = Collapse.
    *   **Downstream Evals:** HellaSwag (10-shot) and LAMBADA (0-shot) to see if "higher rank" actually means "smarter."

## 4. Live Results (Placeholders)

*Training is currently underway. These sections will be updated as checkpoints arrive.*

### A. Learning Rate Sweep
*(Determining the optimal hyperparams for the 300M run)*

| Configuration | Best LR | Sweep Val Loss |
| :--- | :--- | :--- |
| AdamW + QK (Baseline) | `TBD` | `TBD` |
| AdamW No-QK | `TBD` | `TBD` |
| Muon No-QK | `TBD` | `TBD` |
| Muon + QK | `TBD` | `TBD` |

### B. Validation Loss Curves (0 - 300M)
*Does the collapsed model actually perform worse? Or does the model learn to compress intelligence into few dimensions?*

`[PLACEHOLDER: Plot of Val Loss vs Tokens]`

### C. Dimensional Collapse Trajectory
*Tracking the Participation Ratio (PR) of attention keys over time.*

`[PLACEHOLDER: Plot of Mean PR vs Tokens]`

**Current Observation (from 25M Pilot):**
> "At 25M tokens, Muon (No-QK) recovered to PR~51, while AdamW+QK collapsed to PR~16. We expect this gap to widen significantly at 300M."

### D. Downstream Evaluation (Final Checkpoint)
*The ultimate test: Does rank predict reasoning?*

| Config | Seed | HellaSwag | LAMBADA |
| :--- | :--- | :--- | :--- |
| **AdamW + QK-Norm** | 42 | `TBD` | `TBD` |
| **Muon (No QK-Norm)** | 42 | `TBD` | `TBD` |
| *Difference* | | `TBD` | `TBD` |

## 5. What to Watch For

If our hypothesis holds, this experiment could change how we design LLMs. It would suggest that **QK-Normalization is a crutch**—a band-aid that fixes early stability at the cost of long-term intelligence.

By switching to **Muon**, we might be able to remove this constraint entirely, giving future large-scale models (7B, 70B+) significantly more "brain space" (effective dimensionality) without increasing parameter count.

*Check back in ~48 hours for the full logs.*
