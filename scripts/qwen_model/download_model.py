import torch
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

model_name = "Qwen/Qwen2.5-72B-Instruct"
local_path = "/home/s3758869/synchain-absa-emotion/models/Qwen2.5-72B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)
