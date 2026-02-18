from dataclasses import dataclass
from typing import Optional, Tuple
from .llm_config import LLMConfig

@dataclass
class LLMConfig1B(LLMConfig):
    # Model architecture (1.5B Params)
    # Scaled up for 24GB/80GB GPUs
    d_model: int = 2048       
    n_heads: int = 16         # d_k = 128
    n_layers: int = 32
    d_ff: int = 8192         
    
    # GQA parameters
    n_kv_heads: int = 8      
    
    # Training
    batch_size: int = 1      # Maximize params by minimizing local batch
    gradient_accumulation_steps: int = 8 # ~60 steps for 1M tokens
    
    # Data
    max_seq_len: int = 2048  # WARNING: Keep consistent with data prep
    
    # Learning Rates (Need adjustment for model size)
    # Large models usually need lower LRs
    muon_lr: float = 0.012
    adamw_lr: float = 0.003
    warmup_ratio: float = 0.01  # 1% warmup for 1B tokens
    schedule_type: str = "cosine"
    
    # Default tokens for production run
    train_tokens: int = 20000000 # 20 Million tokens
