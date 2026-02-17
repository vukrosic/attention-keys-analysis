import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

def compute_rank_metrics(K, dk):
    """
    Computes spectral metrics for a Key representation matrix K.
    K shape: [tokens, dk]
    """
    # Use float32 for SVD for numerical stability
    K = K.to(torch.float32)
    
    # SVD: K = U Σ V^T
    # We only need singular values
    # Use torch.linalg.svdvals for speed
    try:
        S = torch.linalg.svdvals(K)
    except RuntimeError:
        # Fallback for rare convergence issues
        return None

    # 1. Singular Values
    s = S.detach().cpu().numpy()
    
    # 2. Participation Ratio (Effective Rank)
    # PR = (sum(s))^2 / sum(s^2)
    pr = (np.sum(s)**2) / np.sum(s**2)
    
    # 3. 90% Energy Rank
    # Find min r such that sum(s_i^2 for i=1..r) / sum(s_i^2) >= 0.90
    energies = s**2
    total_energy = np.sum(energies)
    cumulative_energy = np.cumsum(energies) / total_energy
    r90 = np.searchsorted(cumulative_energy, 0.90) + 1
    
    # 4. Condition Number
    cond = s[0] / (s[-1] + 1e-10)
    
    return {
        "participation_ratio": float(pr),
        "r90": int(r90),
        "condition_number": float(cond),
        "spectrum": s.tolist()
    }

class RankProbe:
    """
    Probes a model's internal representations during training/eval.
    """
    def __init__(self, model, dk, device, eval_batch=None):
        self.model = model
        self.dk = dk
        self.device = device
        self.eval_batch = eval_batch
        self.results = {}

    def run_probe(self, step):
        if self.eval_batch is None:
            return
        
        self.model.eval()
        hooks = []
        captured_ks = {}

        def get_hook(layer_idx):
            def hook(module, input, output):
                # Output of k_norm or the raw K projection
                # Expected shape: [batch, seq, heads, dk]
                captured_ks[layer_idx] = output.detach()
            return hook

        # Register hooks on K projections
        for i, block in enumerate(self.model.transformer_blocks):
            # We hook the k_norm if it exists, otherwise we'd need to hook the projection
            # In our implementation, we want to measure the input to the attention mechanism
            # but AFTER positional encoding and norm.
            # In layers.py, Q/K are defined in forward:
            # Q = self.rotary(self.q_norm(Q))
            # K = self.rotary(self.k_norm(K))
            # Let's hook the rotary output for K
            hooks.append(block.attention.rotary.register_forward_hook(get_hook(i)))

        # Run forward pass
        with torch.no_grad():
            x = self.eval_batch.to(self.device)
            _ = self.model(x)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute metrics per layer
        layer_metrics = {}
        for layer_idx, K_BTHD in captured_ks.items():
            # K_BTHD shape: [B, T, H, D]
            B, T, H, D = K_BTHD.shape
            # Combine B and T, but keep heads separate or combined?
            # Let's compute per-head and then average or keep separate.
            # Usually rank is head-specific.
            head_metrics = []
            for h_idx in range(H):
                K_head = K_BTHD[:, :, h_idx, :].reshape(-1, D) # [B*T, D]
                m = compute_rank_metrics(K_head, D)
                if m:
                    head_metrics.append(m)
            
            if head_metrics:
                # Average metrics across heads for this layer
                layer_metrics[layer_idx] = {
                    "pr_avg": float(np.mean([hm["participation_ratio"] for hm in head_metrics])),
                    "r90_avg": float(np.mean([hm["r90"] for hm in head_metrics])),
                    "cond_avg": float(np.mean([hm["condition_number"] for hm in head_metrics])),
                    # We don't store the full spectrum here to save space
                }

        self.results[step] = layer_metrics
        return layer_metrics
