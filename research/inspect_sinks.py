import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from train_llm import prepare_datasets
from torch.utils.data import DataLoader

def inspect_attention_sinks():
    print("🔍 Inspecting for Attention Sinks...")
    
    config = LLMConfig()
    config.compile_model = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinimalLLM(config).to(device)
    model.eval()
    
    # We want to see if there's an index-based bias in Key norms
    # Hook into a middle layer where we saw the outlier spike
    layer_idx = 14
    target_block = model.transformer_blocks[layer_idx]
    
    sink_stats = {
        "index_0_norm": 0,
        "other_indices_norm": 0
    }

    def hook(module, input, output):
        # output: [B, T, H, D]
        with torch.no_grad():
            norms = torch.norm(output, dim=-1) # [B, T, H]
            index_0 = norms[:, 0, :].mean().item()
            others = norms[:, 1:, :].mean().item()
            sink_stats["index_0_norm"] = index_0
            sink_stats["other_indices_norm"] = others
            
            # Also check Max Abs for index 0 vs others
            abs_0 = output[:, 0, :, :].abs().max().item()
            abs_others = output[:, 1:, :, :].abs().max().item()
            sink_stats["index_0_max_abs"] = abs_0
            sink_stats["others_max_abs"] = abs_others

    handle = target_block.attention.k_norm.register_forward_hook(hook)
    
    # Setup minimal data for inspection
    data_cfg = DataConfig(dataset_path="auto", seq_length=config.max_seq_len, num_samples=10, cache_dir="./hf_cache")
    tokenizer = setup_tokenizer(data_cfg)
    train_ds, _ = prepare_datasets(data_cfg, tokenizer)
    loader = DataLoader(train_ds, batch_size=2)
    
    # Get one batch
    batch = next(iter(loader))
    x = batch["input_ids"].to(device)
    
    model(x)
    handle.remove()
    
    print(f"\n📊 Layer {layer_idx} Key Magnitude Distribution:")
    print(f"  - First Token (Index 0) Mean Norm: {sink_stats['index_0_norm']:.4f}")
    print(f"  - Other Tokens Mean Norm:        {sink_stats['other_indices_norm']:.4f}")
    print(f"  - First Token Max Abs Value:     {sink_stats['index_0_max_abs']:.4f}")
    print(f"  - Other Tokens Max Abs Value:    {sink_stats['others_max_abs']:.4f}")
    
    ratio = sink_stats['index_0_norm'] / sink_stats['other_indices_norm']
    if ratio > 1.1:
        print(f"\n✅ CONFIRMED: Token 0 has {((ratio-1)*100):.1f}% more energy than others. Classic Attention Sink detected.")
    else:
        print("\n❓ INCONCLUSIVE: Magnitude is evenly distributed in this snapshot. Sinks might be more dynamic or token-content dependent.")

if __name__ == "__main__":
    inspect_attention_sinks()
