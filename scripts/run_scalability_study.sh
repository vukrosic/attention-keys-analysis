#!/bin/bash
set -e

echo "🚀 Starting Full Research Pipeline 🚀"

# Step 3: Val Set (Already done, but safe to RERUN)
echo "Step 3: Ensuring Fixed Validation Set..."
python research/create_val_set.py

# Step 4: LR Sweep
echo "Step 4: Running Learning Rate Sweep (20 runs x 20M tokens)..."
# This takes time!
python research/run_full_study.py --mode sweep > research_results/scalability_study/sweep.log 2>&1
echo "✅ Sweep Complete. Results in research_results/scalability_study/sweep.log"

# IMPORTANT: You must manually check the log and update LRs in research/run_full_study.py before proceeding!
echo "⚠️  ACTION REQUIRED: Check sweep.log and update 'lrs' dict in research/run_full_study.py (lines ~325)"
echo "Then un-comment the next lines in this script to run production."

# Step 5: Main Runs
# echo "Step 5: Running Production Runs (8 runs x 300M tokens)..."
# python research/run_full_study.py --mode train
# echo "✅ Production Runs Complete."
