from pathlib import Path

import pandas as pd

INPUT_FILES = [
    # "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/final/corona_nlp_test_not_news_proc.csv",
    "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/final/covidsenti_full_not_news_proc.csv"
]

OUTPUT_DIR = Path(
    "/home/s3758869/synchain-absa-emotion/data/input_data/chunks_for_teacher_model_ann"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 2000

for file_path in INPUT_FILES:
    data = pd.read_csv(file_path)

    if "COVID-19-NLP" in file_path:
        file_name = "covid19nlp"
    else:
        file_name = "covidsenti"

    positive = data[
        data["original_sentiment"].str.lower().str.contains("positive")
    ].copy()
    negative = data[
        data["original_sentiment"].str.lower().str.contains("negative")
    ].copy()
    neutral = data[data["original_sentiment"].str.lower() == "neutral"].copy()

    print(f"Processing {file_name}")
    print(f"Total samples: {len(data)}")
    print(f"Sentiment distribution:")
    print(f"  Positive: {len(positive)}")
    print(f"  Negative: {len(negative)}")
    print(f"  Neutral: {len(neutral)}\n")

    samples_per_sentiment = CHUNK_SIZE // 3

    chunk_idx = 0

    pos_idx = 0
    neg_idx = 0
    neu_idx = 0

    while pos_idx < len(positive) or neg_idx < len(negative):
        chunk_parts = []

        if pos_idx < len(positive):
            chunk_parts.append(positive.iloc[pos_idx : pos_idx + samples_per_sentiment])
            pos_idx += samples_per_sentiment

        if neg_idx < len(negative):
            chunk_parts.append(negative.iloc[neg_idx : neg_idx + samples_per_sentiment])
            neg_idx += samples_per_sentiment

        if neu_idx < len(neutral):
            chunk_parts.append(neutral.iloc[neu_idx : neu_idx + samples_per_sentiment])
            neu_idx += samples_per_sentiment

        if chunk_parts:
            chunk = pd.concat(chunk_parts, ignore_index=True)
            chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

            output_file = OUTPUT_DIR / f"{file_name}_chunk{chunk_idx}.csv"
            chunk.to_csv(output_file, index=False)

            print(
                f"Chunk {chunk_idx}: {len(chunk)} samples - {chunk['original_sentiment'].value_counts().to_dict()}"
            )

            chunk_idx += 1

    while neu_idx < len(neutral):
        chunk = neutral.iloc[neu_idx : neu_idx + CHUNK_SIZE].copy()
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

        output_file = OUTPUT_DIR / f"{file_name}_chunk{chunk_idx}.csv"
        chunk.to_csv(output_file, index=False)

        print(
            f"Chunk {chunk_idx}: {len(chunk)} samples - {chunk['original_sentiment'].value_counts().to_dict()}"
        )

        neu_idx += CHUNK_SIZE
        chunk_idx += 1
