# Step-by-Step Research Plan

---

## Step 1: Document Your Architecture

Create a file called `experiment_config.md`. Fill in every field:

```
Model: [GPT-2 / LLaMA-style / other]
Parameters: 80M
Layers: 22
Heads: [?]
Head dim: 64
Vocab size: [?]
Context length: [?]
Dataset: [?]
Tokenizer: [?]
QK-Norm type: [RMSNorm / LayerNorm]
QK-Norm placement: [before RoPE / after RoPE]
Weight decay: [?]
Batch size: [?]
Warmup steps: [?]
LR schedule: [cosine / other]
Gradient clipping: [?]
```

Do not proceed until every field is filled. This file goes in your blog post later.

---

## Step 2: Build Your Logging Code

Before any training, make sure your training loop logs the following **every 500 steps**:

### Must log:
```python
# 1. Training loss (you already have this)
log("train_loss", loss.item(), step)

# 2. Validation loss on fixed held-out set
#    Use ~5M tokens, same set every time, never trained on
if step % 500 == 0:
    val_loss = evaluate(model, val_dataloader)
    log("val_loss", val_loss, step)

# 3. Per-layer Participation Ratio of K projections
#    You already have this from RankProbe
for layer_idx in range(num_layers):
    pr = compute_participation_ratio(model, layer_idx)
    log(f"pr_layer_{layer_idx}", pr, step)

# 4. Full singular value spectrum per layer
#    Save raw singular values to disk every 5000 steps
if step % 5000 == 0:
    for layer_idx in range(num_layers):
        sv = compute_singular_values(model, layer_idx)
        save(f"sv_layer_{layer_idx}_step_{step}.npy", sv)
```

### Should log:
```python
# 5. Mean PR across all layers (quick summary metric)
mean_pr = mean([pr for all layers])
log("mean_pr", mean_pr, step)

# 6. Attention entropy per layer
#    Run a small batch through, capture attention weights,
#    compute entropy of each head's attention distribution, average
for layer_idx in range(num_layers):
    entropy = compute_attention_entropy(model, layer_idx, sample_batch)
    log(f"attn_entropy_layer_{layer_idx}", entropy, step)
```

### Test your logging:
- Run 1000 steps of any config
- Verify you can load and plot every metric
- Fix any bugs now, not during the real runs

---

## Step 3: Create Your Fixed Validation Set

```python
# Take 5M tokens from your dataset
# Save them separately
# Never include them in training data
# Use the SAME validation set for all 12 runs

val_data = load_tokens("your_dataset", split="val", max_tokens=5_000_000)
save("val_set.bin", val_data)
```

This ensures loss comparisons are fair across runs.

---

## Step 4: Learning Rate Sweeps

### 4a: Define sweep ranges

```
AdamW (no QK-Norm):  [1e-4, 3e-4, 6e-4, 1e-3, 2e-3]
AdamW + QK-Norm:     [3e-4, 6e-4, 1e-3, 2e-3, 4e-3]
Muon (no QK-Norm):   [adapt to Muon's typical range, 5 values]
Muon + QK-Norm:      [same Muon range, possibly shifted higher]
```

### 4b: Run short sweeps

```
4 configs × 5 LRs = 20 short runs
Each run: 20M tokens only
Log validation loss at end
```

### 4c: Pick winners

For each of the 4 configs, pick the LR with the lowest validation loss at 20M tokens.

### 4d: Save this as a plot

```python
# For each config, plot LR (x-axis, log scale) vs val_loss (y-axis)
# Mark the chosen LR with a star
# This plot goes in your blog post
```

Write down your 4 chosen learning rates:

```
AdamW:           lr = [chosen]
AdamW + QK-Norm: lr = [chosen]
Muon:            lr = [chosen]
Muon + QK-Norm:  lr = [chosen]
```

---

## Step 5: Main Training Runs

### 4 configs × 2 seeds = 8 runs

```
Run 1:  AdamW,           seed=42
Run 2:  AdamW,           seed=137
Run 3:  AdamW + QK-Norm, seed=42
Run 4:  AdamW + QK-Norm, seed=137
Run 5:  Muon,            seed=42
Run 6:  Muon,            seed=137
Run 7:  Muon + QK-Norm,  seed=42
Run 8:  Muon + QK-Norm,  seed=137
```

Each run:
- 300M tokens
- Best LR from Step 4
- All logging from Step 2 active
- Save checkpoints at 50M, 100M, 150M, 200M, 250M, 300M tokens

### If you truly only do 1 seed:
Runs 1, 3, 5, 7 only. But you lose all ability to claim reproducibility.

---

## Step 6: Spot Check During Training

At each checkpoint, do a quick sanity look:

```
At 50M tokens:  Do loss curves look reasonable? Any NaNs?
At 100M tokens: Does PR trajectory match your earlier observations?
At 200M tokens: Are any runs clearly broken?
```

Don't change anything. Just look. If a run produces NaNs, note the step and restart from the last good checkpoint if possible.

---

## Step 7: Run Downstream Evals

After all runs finish at 300M tokens, evaluate each of the 8 final checkpoints on:

```
1. Held-out WikiText perplexity (or your dataset's validation perplexity)
   - You already have this from val_loss logging

2. HellaSwag (10-shot)
   - Use lm-evaluation-harness or equivalent
   
3. LAMBADA (0-shot, last word accuracy)
   - Use lm-evaluation-harness
```

Record results in a table:

```
| Config           | Seed | Val PPL | HellaSwag | LAMBADA |
|------------------|------|---------|-----------|---------|
| AdamW            | 42   |         |           |         |
| AdamW            | 137  |         |           |         |
| AdamW + QK-Norm  | 42   |         |           |         |
| ...              |      |         |           |         |
```

---

## Step 8: Generate Plots

### Plot 1 — Validation Loss Curves (MOST IMPORTANT)

```
X-axis: tokens (0 to 300M)
Y-axis: validation loss
4 lines, one per config
If 2 seeds: show mean line with light shading for the two individual runs
Title: "Validation Loss"
```

### Plot 2 — Mean PR Trajectory

```
X-axis: tokens (0 to 300M)
Y-axis: mean participation ratio across all layers
4 lines, one per config
Same shading treatment as Plot 1
Title: "Mean Participation Ratio Over Training"
```

### Plot 3 — Loss vs PR Scatter

```
X-axis: mean PR
Y-axis: validation loss
Each point = one run at one checkpoint (sample every 50M tokens)
Color by config
This answers: "Does higher PR mean lower loss?"
```

### Plot 4 — Per-Layer PR Heatmap at 300M Tokens

```
Rows: layer 1 to 22
Columns: 4 configs
Cell value: PR (averaged across seeds)
Color scale: red (low/collapsed) to blue (high/healthy)
```

### Plot 5 — Singular Value Spectra

```
Pick layer 1, layer 11, layer 22 (early, middle, late)
For each layer: 4 lines showing singular values sorted descending
One subplot per layer
This shows WHAT the collapse looks like, not just a number
```

### Plot 6 — LR Sweep Results

```
4 subplots, one per config
X-axis: learning rate (log scale)
Y-axis: val loss at 20M tokens
Star on the chosen LR
```

---

## Step 9: Build the Results Table

```
| Config          | Best LR | Val Loss | Mean PR | HellaSwag | LAMBADA |
|-----------------|---------|----------|---------|-----------|---------|
| AdamW           |         |          |         |           |         |
| AdamW + QK-Norm |         |          |         |           |         |
| Muon            |         |          |         |           |         |
| Muon + QK-Norm  |         |          |         |           |         |
```

If 2 seeds, show mean ± difference.

---

## Step 10: Determine Your Conclusion

Look at your data honestly. Which scenario are you in?

**Scenario A: Muon-no-QKNorm wins on BOTH loss AND PR**
→ Strong result. Your headline: "Muon provides implicit attention logit control, making QK-Norm unnecessary and harmful."

**Scenario B: Muon-no-QKNorm wins on PR but ties or loses on loss**
→ Moderate result. Your headline: "QK-Norm causes measurable dimensional collapse but the impact on performance is limited at this scale. Open question at larger scale."

**Scenario C: AdamW+QKNorm wins on loss despite collapsed PR**
→ Surprising and interesting. Your headline: "High-dimensional representations may be unnecessary. Models can perform well with collapsed attention geometry."

**Scenario D: Results are noisy or inconsistent across seeds**
→ Don't post. Go to 3+ seeds or change approach.

Write your conclusion AFTER looking at the data. Not before.

---

## Step 11: Write the Blog Post

### Structure:

```
Title: [Based on what you actually found — decide in Step 10]

1. TLDR (3 sentences max)

2. Experimental Setup
   - Architecture details (paste from experiment_config.md)
   - The 4 configs and why
   - LR tuning methodology (with Plot 6)
   - "2 seeds per config, 300M tokens, 80M param model"

3. Results: Loss (Plot 1)
   - Lead with what people care about: which config trains best?

4. Results: Representation Quality (Plots 2, 4, 5)
   - PR trajectories
   - Per-layer breakdown
   - Singular value spectra

5. The Relationship Between Loss and PR (Plot 3)
   - Does PR predict performance? Show the scatter.

6. Downstream Evaluation (Results table)
   - Do the loss differences translate to eval differences?

7. Discussion
   - Your mechanistic explanation
   - Why you think this happens

8. Limitations (BE HONEST)
   - 80M params, single architecture
   - 2 seeds
   - Unknown whether this holds at larger scale
   - QK-Norm works in production models (Gemma 2, etc.)

9. Recommendations
   - Practical advice for practitioners
```

---

## Step 12: Self-Review Before Posting

Go through this checklist:

```
[ ] Every claim is backed by a plot or number
[ ] LR was tuned independently per config
[ ] Loss curves are shown (not just PR)
[ ] Limitations section is honest
[ ] I haven't used the word "proves"
[ ] Title matches what I actually found
[ ] Someone who disagrees with me would still say the experiment was fair
```

---

## Total Effort Estimate

```
Step 1:           1 hour
Step 2:           3-4 hours
Step 3:           30 minutes
Step 4:           1 day compute + 2 hours analysis
Step 5:           1-2 days compute
Step 6:           Passive monitoring
Step 7:           2-3 hours
Steps 8-9:        1 day
Steps 10-12:      1-2 days writing
                  ─────────────
Total:            ~7-10 days
```
