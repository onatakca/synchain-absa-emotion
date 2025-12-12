import argparse
import sys

import pandas as pd
import spacy
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


def preprocess_dataset(input_path, output_path, sample_size=None):
    print(f"Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Reading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original dataset size: {len(df)} tweets")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    if sample_size is not None and sample_size < len(df):
        print(f"Sampling {sample_size} tweets...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    df["conllu_parse"] = ""

    print(f"Parsing {len(df)} tweets...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing tweets"):
        tweet_text = str(row["tweet"])
        conllu = parse_to_conllu(tweet_text, nlp)
        df.at[idx, "conllu_parse"] = conllu

    df_parsed = df[df["conllu_parse"] != ""].copy()
    print(f"\nSuccessfully parsed: {len(df_parsed)}/{len(df)} tweets")

    if len(df_parsed) < len(df):
        print(f"Failed to parse: {len(df) - len(df_parsed)} tweets")

    print(f"Saving results to {output_path}...")
    df_parsed.to_csv(output_path, index=False)

    print(f"\n✓ Done! Processed dataset saved to {output_path}")
    print(f"  Total tweets: {len(df_parsed)}")
    print(f"  Columns: {df_parsed.columns.tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess COVIDSenti dataset with syntactic parsing"
    )
    parser.add_argument(
        "--input", type=str, default="COVIDSenti.csv", help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="COVIDSenti_parsed.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.sample)


if __name__ == "__main__":
    main()
