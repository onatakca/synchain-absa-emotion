from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scripts.modeling.dataset import ABSADataset, get_data, Collator
from scripts.modeling.evaluate import EvaluationCallback
import json
from datetime import datetime
from pathlib import Path
import os
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

TRAIN_FILES = [
    "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk1_annotated.json",
    "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk2_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk3_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk4_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk5_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk6_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk7_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk8_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk9_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk10_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk11_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covid19nlp_chunk0_annotated.json"
]

VALIDATION_FILES = [
    "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk0_annotated.json",
]

config_file_path = "/home/s3758869/synchain-absa-emotion/scripts/modeling/configs/qwen7b/full_pipeline_qwen7b_all_steps.json"

with open(config_file_path, "r") as f:
    config = json.load(f)
    
train_dataset = ABSADataset(get_data(TRAIN_FILES), tasks = config["tasks"], max_samples = config["max_samples"])
val_dataset = ABSADataset(get_data(VALIDATION_FILES), tasks=  config["tasks"], max_samples= config["max_samples"])


run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = config["model_identifier"].split("/")[-1]
base_out = Path(config["out_path"]) / model_name

run_dir = base_out / run_id
ckpt_dir = run_dir / "checkpoints"
logs_dir = run_dir / "logs"
eval_dir = run_dir / "eval"

for p in [ckpt_dir, logs_dir, eval_dir]:
    p.mkdir(parents=True, exist_ok=True)

with open(run_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2, sort_keys=True)
    
student_tokenizer = AutoTokenizer.from_pretrained(
    config["model_identifier"],
    local_files_only=True,
    padding_side='left'
    )
collator = Collator(tokenizer=student_tokenizer, max_length=config["max_length"])

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    load_in_4bit=False,    
)

student_model = AutoModelForCausalLM.from_pretrained(
    config["model_identifier"],
    local_files_only=True,
    torch_dtype=torch.bfloat16 if config.get("bf16", False) else torch.float32,
    quantization_config=bnb_config,
    device_map="auto",  
)

student_model.config.use_cache = False 
student_model = prepare_model_for_kbit_training(student_model)

if config.get("gradient_checkpointing", False):
    student_model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

student_model = get_peft_model(student_model, lora_config)
student_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=ckpt_dir,
    logging_dir = logs_dir,
    per_device_train_batch_size=config["train_batch_size"],
    per_device_eval_batch_size=config["val_batch_size"],
    learning_rate=config["learning_rate"],
    remove_unused_columns=False,
    logging_strategy="steps",
    logging_steps=config["logging_steps"],
    logging_first_step=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    dataloader_num_workers=min(os.cpu_count(),config["dataloader_num_workers"]),
    eval_strategy="no",
    bf16=config.get("bf16", False),
    optim="paged_adamw_8bit", 
    gradient_checkpointing=config.get("gradient_checkpointing", False),
    save_strategy="steps",
    save_steps=config["save_steps"],
    save_total_limit=config["save_total_limit"],
    max_steps=config["max_steps"],
    seed=config["seed"],
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,    
    data_collator=collator, 
)

print("Training")
trainer.train()

trainer.save_state()
trainer.save_model(str(run_dir / "final_model"))
student_tokenizer.save_pretrained(str(run_dir / "final_model"))

print("Getting train loss")
final_metrics = trainer.evaluate()
with open(eval_dir / "eval_after_train.json", "w") as f:
    json.dump(final_metrics, f, indent=2, sort_keys=True)
    f.flush()

print("Evaluation callback :")
eval_callback = EvaluationCallback(val_dataset, student_tokenizer, config["max_new_tokens"], config["tasks"])
gen_metrics = eval_callback.on_evaluate(training_args, trainer.state, trainer.control, model=student_model)
with open(eval_dir / "gen_eval_after_train.json", "w") as f:
    json.dump(gen_metrics, f, indent=2, sort_keys=True)
    f.flush()
