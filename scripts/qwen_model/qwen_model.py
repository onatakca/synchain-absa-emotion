import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

LOCAL_MODEL_CACHE_DIR = (
    "/home/s3758869/synchain-absa-emotion/models/Qwen2.5-72B-Instruct"
)


def load_model(
    model_name="Qwen/Qwen2.5-72B-Instruct",
    quantization_bits=4,
    device_map="auto",
    cache_dir="./",
):
    if cache_dir is None:
        raise Exception("Give proper model cache dir")

    if quantization_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(cache_dir)) if Path(cache_dir).exists() else model_name,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2ForCausalLM.from_pretrained(
        model_identifier,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    
    try:
        model.gradient_checkpointing_enable()
        print("Enabled checkpointing")
    except:
        pass
    
    print(f"Model loaded successfully!")
    print(f"Device map: {model.hf_device_map}")

    return model, tokenizer


def generate_response(
    model, tokenizer, messages, max_new_tokens=512, temperature=0.7, do_sample=True
):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size=2):
    if prompts and isinstance(prompts[0], list):
        texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in prompts
        ]
    else:
        texts = prompts

    all_decoded = []

    with tqdm(total=len(texts), desc="Generating", unit="batch") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_decoded.extend(decoded)

            del inputs, outputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            pbar.update(len(batch_texts))

    return all_decoded


def generate_batch_with_checkpoint(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    batch_size,
    checkpoint_file,
    checkpoint_interval=10,
):
    checkpoint_path = Path(checkpoint_file)

    if prompts and isinstance(prompts[0], list) and isinstance(prompts[0][0], dict):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in prompts
        ]
    else:
        texts = prompts

    total = len(texts)

    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            saved_outputs = json.load(f)
        if len(saved_outputs) > total:
            raise ValueError(
                f"Checkpoint has more outputs ({len(saved_outputs)}) than prompts ({total})."
            )
        start_idx = len(saved_outputs)
        all_decoded = saved_outputs
        print(f"[{checkpoint_path.name}] Resuming from {start_idx}/{total}")
    else:
        all_decoded = []
        start_idx = 0
        print(f"[{checkpoint_path.name}] No checkpoint found, starting fresh.")

    if start_idx >= total:
        print(f"[{checkpoint_path.name}] All {total} prompts already processed.")
        return all_decoded

    batch_counter = 0
    with tqdm(
        total=total - start_idx,
        desc=f"Generating ({checkpoint_path.name})",
        unit=f"batch",
    ) as pbar:
        for i in range(start_idx, total, batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]

            decoded = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            all_decoded.extend(decoded)
            batch_counter += 1

            if batch_counter % checkpoint_interval == 0 or i + batch_size >= total:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(all_decoded, f, indent=2, ensure_ascii=False)

            del inputs, outputs, generated_ids
            
            # More aggressive memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            pbar.update(len(batch_texts))

    return all_decoded
