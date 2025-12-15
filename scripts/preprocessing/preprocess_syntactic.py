import sys

import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm


def token_to_conllu(token, idx):
    head_idx = token.head.i - token.sent[0].i + 1 if token.head.i != token.i else 0

    return "\t".join(
        [
            str(idx),
            token.text,
            token.lemma_,
            token.pos_,
            token.tag_,
            "_",
            str(head_idx),
            token.dep_,
            "_",
            "_",
        ]
    )


def parse_to_conllu(text, nlp):
    try:
        doc = nlp(text)

        conllu_lines = []
        for sent in doc.sents:
            conllu_lines.append(f"# text = {sent.text}")

            for idx, token in enumerate(sent, start=1):
                conllu_lines.append(token_to_conllu(token, idx))

            conllu_lines.append("")

        return "\n".join(conllu_lines).strip()

    except Exception as e:
        print(f"Error parsing text: {text[:50]}... - {e}", file=sys.stderr)
        return ""


def preprocess_dataset(input_path, output_path, tweet_column, sample_size=None):
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(input_path)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    df["conllu_parse"] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing"):
        tweet_text = str(row[tweet_column])
        conllu = parse_to_conllu(tweet_text, nlp)
        df.at[idx, "conllu_parse"] = conllu

    df_parsed = df[df["conllu_parse"] != ""].copy()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_parsed.to_csv(output_path, index=False)

    print(f"Parsed {len(df_parsed)}/{len(df)} tweets → {output_path}")


def main():
    datasets = [
        {
            "input": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/processed/corona_nlp_test_not_news_proc.csv",
            "output": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/final/corona_nlp_test_not_news_proc.csv",
            "tweet_column": "tweet",
        },
        {
            "input": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/processed/corona_nlp_train_not_news_proc.csv",
            "output": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/final/corona_nlp_train_not_news_proc.csv",
            "tweet_column": "tweet",
        },
        {
            "input": "/home/s3758869/synchain-absa-emotion/data/input_data/SenWave/processed/senwave_not_news_proc.csv",
            "output": "/home/s3758869/synchain-absa-emotion/data/input_data/SenWave/final/senwave_not_news_proc.csv",
            "tweet_column": "tweet",
        },
        {
            "input": "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/processed/covidsenti_full_not_news_proc.csv",
            "output": "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/final/covidsenti_full_not_news_proc.csv",
            "tweet_column": "tweet",
        },
    ]

    for dataset in datasets:
        print(f"\nProcessing: {dataset['input']}")
        preprocess_dataset(dataset["input"], dataset["output"], dataset["tweet_column"])


if __name__ == "__main__":
    main()
