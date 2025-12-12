#!/usr/bin/env python3
"""
Pre-download Qwen2.5-72B model on login node (which has internet access).
Compute nodes will then use the cached version.
"""

from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import torch

model_name = "Qwen/Qwen2.5-72B-Instruct"

print("=" * 60)
print("DOWNLOADING QWEN2.5-72B-INSTRUCT")
print("=" * 60)
print(f"Model: {model_name}")
print(f"Cache location: ~/.cache/huggingface/")
print()
print("This will download ~40GB and may take 20-30 minutes...")
print()

# Download tokenizer
print("1/2 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
print("✓ Tokenizer downloaded")
print(f"  Vocab size: {len(tokenizer)}")
print()

# Download model weights (without loading to GPU)
print("2/2 Downloading model weights...")
print("(This is the large download - ~40GB)")
print("Progress will show below...\n")

model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu",  # Don't use GPU, just download
    low_cpu_mem_usage=True,
)
print("\n✓ Model weights downloaded")
print(f"  Parameters: {model.num_parameters():,}")
print()

print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print("The model is now cached locally.")
print("Compute nodes can use it without internet access.")
print()
print("Next step: sbatch scripts/run_annotation.sh")
