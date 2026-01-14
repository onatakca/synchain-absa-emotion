from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from scripts.modeling.dataset import ABSADataset, get_data, Collator
from scripts.modeling.evaluate import EvaluationCallback
import json
from datetime import datetime
from pathlib import Path
import os

TRAIN_FILES = [
    "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk1_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk2_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk3_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk4_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk5_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk6_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk7_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk8_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk9_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk10_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk11_annotated.json",
]

VALIDATION_FILES = [
    "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk0_annotated.json",
    # "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covid19nlp_chunk0_annotated.json"
]

config_file_path = "/home/s3758869/synchain-absa-emotion/scripts/modeling/configs/full_pipeline.json"

with open(config_file_path, "r") as f:
    config = json.load(f)
    
train_dataset = ABSADataset(get_data(TRAIN_FILES), tasks = config["tasks"])
val_dataset = ABSADataset(get_data(VALIDATION_FILES), tasks=  config["tasks"])

student_tokenizer = AutoTokenizer.from_pretrained(
    config["model_identifier"],
    local_files_only=True
    )

student_model = AutoModelForCausalLM.from_pretrained(
    config["model_identifier"],
    local_files_only=True
    )

collator = Collator(tokenizer=student_tokenizer, max_length=config["max_length"])

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
    

training_args = TrainingArguments(
    output_dir=ckpt_dir,
    logging_dir = logs_dir,
    per_device_train_batch_size=config["train_batch_size"],
    per_device_eval_batch_size=config["val_batch_size"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    remove_unused_columns=False,
    logging_strategy="steps",
    logging_steps=config["logging_steps"],
    logging_first_step=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    dataloader_num_workers=min(os.cpu_count(),config["dataloader_num_workers"]),
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=config["save_steps"],
    save_total_limit=config["save_total_limit"],
    seed=config["seed"],
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,    
    data_collator=collator, 
    callbacks=[EvaluationCallback(val_dataset, student_tokenizer, config["max_new_tokens"], config["tasks"])] 
)

eval_metrics = trainer.evaluate()
with open(eval_dir / "eval_before_train.json", "w") as f:
    json.dump(eval_metrics, f, indent=2, sort_keys=True)
    
trainer.train()

final_metrics = trainer.evaluate()
with open(eval_dir / "eval_after_train.json", "w") as f:
    json.dump(final_metrics, f, indent=2, sort_keys=True)
    
trainer.save_model(str(run_dir / "final_model"))
student_tokenizer.save_pretrained(str(run_dir / "final_model"))