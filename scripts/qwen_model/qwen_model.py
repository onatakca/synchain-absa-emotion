import os

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
        cache_dir,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
        padding_side='left',
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2ForCausalLM.from_pretrained(
        cache_dir,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

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
    
    with tqdm(total=len(texts), desc="Generating", unit="prompt") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_decoded.extend(decoded)
            
            del inputs, outputs, generated_ids
            torch.cuda.empty_cache()
            
            pbar.update(len(batch_texts))
    
    return all_decoded
