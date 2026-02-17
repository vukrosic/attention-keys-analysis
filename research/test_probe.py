import torch
import sys
import os
sys.path.append(os.getcwd())
from models.llm import MinimalLLM
from configs.llm_config import LLMConfig
from research.svd_probe import RankProbe

def test_probe():
    config = LLMConfig()
    config.n_layers = 1 # Small for test
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Random input
    x = torch.randint(0, config.vocab_size, (1, 1024)).to(device)
    
    probe = RankProbe(model, config.d_model // config.n_heads, device, eval_batch=x)
    print("Running probe...")
    results = probe.run_probe(0)
    print("Results:", results)

if __name__ == "__main__":
    test_probe()
