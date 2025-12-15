import json
from pathlib import Path

import pandas as pd

from scripts.annotation.parsing import extract_aspects, is_news_like
from scripts.qwen_model.prompts import (prompt_aspect_extraction,
                                        prompt_emotion_classification,
                                        prompt_opinion_extraction,
                                        prompt_sentiment_classification,
                                        prompt_syntactic_parsing)

from scripts.qwen_model.qwen_model import generate_batch, load_model, generate_batch_with_checkpoint

N_SAMPLES = 50
INPUT_FILE = "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/final/COVIDSenti_full_parsed.csv"
OUTPUT_FILE = f"/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-72b-instruct_annotation/COVIDSenti_conversational_annotated_{N_SAMPLES}.json"
CHECKPOINT_DIR = "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-72b-instruct_annotation/checkpoints"

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
MODEL_DIR = "/home/s3758869/synchain-absa-emotion/models/Qwen2.5-72B-Instruct"
QUANTS = 4
BATCH_SIZE = 8

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

def save_checkpoint(data, stage, filename):
    checkpoint_path = Path(CHECKPOINT_DIR) / filename
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved: {stage} -> {checkpoint_path}")

df = pd.read_csv(INPUT_FILE)
df["is_news"] = df["tweet"].apply(is_news_like)
df = df[~df["is_news"]].head(N_SAMPLES).copy()

tweets = df["tweet"].tolist()
conllus = df["conllu_parse"].tolist()


model, tokenizer = load_model(
    MODEL_NAME, QUANTS, device_map="auto", cache_dir=MODEL_DIR
)

prompts_r1 = [prompt_aspect_extraction(t, c) for t, c in zip(tweets, conllus)]

r1_ckpt = f"{CHECKPOINT_DIR}/r1_aspects_{N_SAMPLES}.json"
r1_outputs = generate_batch_with_checkpoint(
    model,
    tokenizer,
    prompts_r1,
    max_new_tokens=150,
    batch_size=BATCH_SIZE,
    checkpoint_file=r1_ckpt,
)

aspect_lists = [extract_aspects(x) for x in r1_outputs]


tasks = []
for i, (tweet, conllu, aspects) in enumerate(zip(tweets, conllus, aspect_lists)):
    for a in aspects:
        tasks.append((i, tweet, conllu, a))

print(f"Total aspect instances: {len(tasks)}")

prompts_r2 = [
    prompt_syntactic_parsing(tweet, aspect, conllu)
    for (_, tweet, conllu, aspect) in tasks
]

r2_ckpt = f"{CHECKPOINT_DIR}/r2_syntactic_{N_SAMPLES}.json"
r2_outputs = generate_batch_with_checkpoint(
    model,
    tokenizer,
    prompts_r2,
    max_new_tokens=200,
    batch_size=BATCH_SIZE,
    checkpoint_file=r2_ckpt,
)


prompts_r3 = [
    prompt_opinion_extraction(tweet, aspect, syn)
    for (syn, (_, tweet, _, aspect)) in zip(r2_outputs, tasks)
]

r3_ckpt = f"{CHECKPOINT_DIR}/r3_opinions_{N_SAMPLES}.json"
r3_outputs = generate_batch_with_checkpoint(
    model,
    tokenizer,
    prompts_r3,
    max_new_tokens=120,
    batch_size=BATCH_SIZE,
    checkpoint_file=r3_ckpt,
)

prompts_r4 = [
    prompt_sentiment_classification(tweet, aspect, op)
    for (op, (_, tweet, _, aspect)) in zip(r3_outputs, tasks)
]

r4_ckpt = f"{CHECKPOINT_DIR}/r4_sentiments_{N_SAMPLES}.json"
r4_outputs = generate_batch_with_checkpoint(
    model,
    tokenizer,
    prompts_r4,
    max_new_tokens=120,
    batch_size=BATCH_SIZE,
    checkpoint_file=r4_ckpt,
)


prompts_r5 = [
    prompt_emotion_classification(tweet, aspect, op)
    for (op, (_, tweet, _, aspect)) in zip(r3_outputs, tasks)
]

r5_ckpt = f"{CHECKPOINT_DIR}/r5_emotions_{N_SAMPLES}.json"
r5_outputs = generate_batch_with_checkpoint(
    model,
    tokenizer,
    prompts_r5,
    max_new_tokens=120,
    batch_size=BATCH_SIZE,
    checkpoint_file=r5_ckpt,
)


output_data = {}

for i, tweet in enumerate(tweets):
    aspects_dict = {}
    aspect_sentiments = {}
    aspect_syntactic = {}
    aspect_opinions = {}
    aspect_emotions = {}

    aspect_id = 0
    for idx, (task, syn, op, sent, emo) in enumerate(
        zip(tasks, r2_outputs, r3_outputs, r4_outputs, r5_outputs)
    ):
        tweet_idx, _, _, aspect = task
        if tweet_idx == i:
            aspects_dict[aspect] = aspect_id
            aspect_sentiments[aspect_id] = sent
            aspect_syntactic[aspect_id] = syn
            aspect_opinions[aspect_id] = op
            aspect_emotions[aspect_id] = emo
            aspect_id += 1

    output_data[tweet] = {
        "aspects": aspects_dict,
        "aspect_sentiments": aspect_sentiments,
        "aspect_syntactic": aspect_syntactic,
        "aspect_opinions": aspect_opinions,
        "aspect_emotions": aspect_emotions,
    }

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("Saved:", OUTPUT_FILE)
