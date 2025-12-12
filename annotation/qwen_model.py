"""
Quantized Qwen2.5-72B-Instruct model loader for teacher annotation.
"""

import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


class QwenTeacher:
    """Loads and runs quantized Qwen2.5-72B-Instruct for reasoning trace generation."""

    def __init__(self, model_name="Qwen/Qwen2.5-72B-Instruct", device_map="auto",
                 cache_dir=None):
        """
        Load Qwen2.5-72B-Instruct with 4-bit quantization.

        Args:
            model_name: HuggingFace model ID
            device_map: Device mapping strategy (auto uses all available GPUs)
            cache_dir: Directory with cached model files (default: ~/.cache/huggingface)
        """
        # Find the local cached model path
        if cache_dir is None:
            # HuggingFace cache structure
            hub_cache = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_name = model_name.replace("/", "--")
            model_cache_path = os.path.join(hub_cache, f"models--{model_cache_name}")

            # Find the snapshot directory
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                # Get the latest snapshot
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    snapshot = snapshots[0]  # Use first/only snapshot
                    model_path = os.path.join(snapshots_dir, snapshot)
                else:
                    raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
            else:
                raise FileNotFoundError(f"Model cache not found at {model_cache_path}")
        else:
            model_path = cache_dir

        print(f"Loading {model_name} with 4-bit quantization...")
        print(f"Using local model path: {model_path}")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model and tokenizer from local path (no internet needed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

        print(f"Model loaded successfully!")
        print(f"Device map: {self.model.hf_device_map}")

    def generate(self, messages, max_new_tokens=200, temperature=0.7):
        """
        Generate response from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text string
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def generate_reasoning_trace(self, prompt_messages, max_new_tokens=200):
        """
        Generate a single reasoning trace.

        Args:
            prompt_messages: Messages from prompt generation function
            max_new_tokens: Max tokens for this trace

        Returns:
            Reasoning trace string
        """
        return self.generate(prompt_messages, max_new_tokens=max_new_tokens)
