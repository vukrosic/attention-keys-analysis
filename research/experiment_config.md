
Model: MinimalLLM (Custom Dense Transformer)
Parameters: 88M
Layers: 22
Heads: 8
Head dim: 64
Vocab size: 49152
Context length: 2048
Dataset: FineWeb-Edu + Cosmopedia Mix (HuggingFaceTB/smollm-corpus variants)
Tokenizer: HuggingFaceTB/SmolLM2-135M
QK-Norm type: RMSNorm
QK-Norm placement: before RoPE
Weight decay: 0.2
Batch size: 2 (per device)
Warmup steps: 0
LR schedule: constant
Gradient clipping: 1.0
