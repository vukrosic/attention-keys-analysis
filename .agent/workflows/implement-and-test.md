---
description: An autonomous pipeline for exhaustive research exploration, reasoning, and scaling.
---

# Autonomous Research Pipeline

This workflow is an infinite loop designed to find the Global Maximum of performance on a benchmark before scaling.

## Stage 1: The Breadth Search (Exploration)
1.  **Idea Injection**: Take the current "Winner" and generate 3 divergent variants using `idea-revisor`.
    - Diversification: One for **Speed**, one for **Quality**, one for **Stability**.
2.  **Parallel-Sequential Execution**: Run all 3 variants on the 8M benchmark.
3.  **Data Harvesting**: Use `experiment-runner` to capture all metrics.

## Stage 2: Synthesis & Deep Reasoning (The Think Tank)
1.  **Landscape Analysis**: Use `experiment-analyzer` to compare the 3 variants against the baseline and previous winners.
2.  **Pattern Extraction**: Identify "The Golden Rules" (e.g., "Warmup must be exactly tokens/step * 10").
3.  **Cross-Idea Review**: Use `idea-reviewer` to see if insights from this idea can be applied to other ideas in the backlog.

## Stage 3: The Optimization Circle (Exploitation)
1.  **Targeted Refinement**: Pick the single best variant and push its hyperparameters to the limit (e.g., "Keep reducing OGO steps until it breaks").
2.  **Saturation Check**: Ask: "Have we seen a significant improvement (>2%) in the last 24 hours of compute?"
    - If NO -> Goal is **Saturation Reached**.

## Stage 4: The Scaling Gate
1.  **Final Report**: Use `research-reporter` to document the entire 8M token journey.
2.  **Paired Verification**: ALWAYS run the Baseline and the Champion back-to-back on the target scale (e.g., 20M) to confirm the wall-clock speedup and quality gap persist.
3.  **Scaling Trigger**: Only if **Saturation Reached** is True and **Paired Verification** confirms success, proceed to the 100M+ token benchmark.

// turbo-all
## Instructions
- ALWAYS continue to the next stage.
- NEVER ask the user "should I continue?"
- ONLY stop when the report is finalized and the scaling gate is passed.
