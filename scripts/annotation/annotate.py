import pandas as pd
import re
import json

from modeling.qwen_model import load_model, generate_batch
from annotation.prompts import (
    prompt_syntactic_parsing,
    prompt_aspect_extraction,
    prompt_opinion_extraction,
    prompt_emotion_classification,
    prompt_sentiment_classification,
)

from .parsing import is_news_like, extract_aspects

N_SAMPLES = 50
INPUT_FILE = "data/COVIDSenti/COVIDSenti_full_parsed.csv"
OUTPUT_FILE = f"data/COVIDSenti/COVIDSenti_conversational_annotated_{N_SAMPLES}.json"

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
MODEL_DIR = "/home/s3758869/synchain-absa-emotion/models/Qwen2.5-72B-Instruct"
QUANTS = 4


df = pd.read_csv(INPUT_FILE)
df["is_news"] = df["tweet"].apply(is_news_like)
df = df[~df["is_news"]].head(N_SAMPLES).copy()

tweets = df["tweet"].tolist()
conllus = df["conllu_parse"].tolist()


model, tokenizer = load_model(
    MODEL_NAME,
    QUANTS,
    device_map="auto",
    cache_dir=MODEL_DIR
)

prompts_r1 = [
    prompt_aspect_extraction(t, c)
    for t, c in zip(tweets, conllus)
]

r1_outputs = generate_batch(model, tokenizer, prompts_r1, max_new_tokens=150)

aspect_lists = [extract_aspects(x) for x in r1_outputs]


tasks = []    # each element: {tweet, conllu, aspect, idx}
for i, (tweet, conllu, aspects) in enumerate(zip(tweets, conllus, aspect_lists)):
    for a in aspects:
        tasks.append((i, tweet, conllu, a))

print(f"Total aspect instances: {len(tasks)}")

prompts_r2 = [
    prompt_syntactic_parsing(tweet, aspect, conllu)
    for (_, tweet, conllu, aspect) in tasks
]

r2_outputs = generate_batch(model, tokenizer, prompts_r2, max_new_tokens=200)


prompts_r3 = [
    prompt_opinion_extraction(tweet, aspect, syn)
    for (syn, (_, tweet, _, aspect)) in zip(r2_outputs, tasks)
]

r3_outputs = generate_batch(model, tokenizer, prompts_r3, max_new_tokens=120)

prompts_r4 = [
    prompt_sentiment_classification(tweet, aspect, op)
    for (op, (_, tweet, _, aspect)) in zip(r3_outputs, tasks)
]

r4_outputs = generate_batch(model, tokenizer, prompts_r4, max_new_tokens=120)



prompts_r5 = [
    prompt_emotion_classification(tweet, aspect, op)
    for (op, (_, tweet, _, aspect)) in zip(r3_outputs, tasks)
]

r5_outputs = generate_batch(model, tokenizer, prompts_r5, max_new_tokens=120)


output_data = {}

for i, tweet in enumerate(tweets):
    aspects_dict = {}
    aspect_sentiments = {}
    aspect_syntactic = {}
    aspect_opinions = {}
    aspect_emotions = {}
    
    aspect_id = 0
    for idx, (task, syn, op, sent, emo) in enumerate(zip(tasks, r2_outputs, r3_outputs, r4_outputs, r5_outputs)):
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
        "aspect_emotions": aspect_emotions
    }

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("Saved:", OUTPUT_FILE)
