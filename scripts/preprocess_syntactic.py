"""
Syntactic Parsing Preprocessing Script

This script:
1. Loads the COVIDSenti dataset
2. Parses each tweet using spaCy
3. Converts dependency parse to CoNLL-U format
4. Saves the results for downstream processing

Usage:
    python preprocess_syntactic.py --input COVIDSenti.csv --output COVIDSenti_parsed.csv --sample 1000
"""

import argparse
import pandas as pd
import spacy
from tqdm import tqdm
import sys


def token_to_conllu(token, idx):
    """
    Convert a spaCy token to CoNLL-U format string.

    CoNLL-U format columns:
    1. ID: Word index (starting at 1)
    2. FORM: Word form or punctuation symbol
    3. LEMMA: Lemma or stem of word form
    4. UPOS: Universal part-of-speech tag
    5. XPOS: Language-specific part-of-speech tag
    6. FEATS: List of morphological features (left blank)
    7. HEAD: Head of the current word (0 if root)
    8. DEPREL: Universal dependency relation to the HEAD
    9. DEPS: Enhanced dependency graph (left blank)
    10. MISC: Any other annotation (left blank)
    """
    # Handle cases where head points to itself (should be 0 for root)
    head_idx = token.head.i - token.sent[0].i + 1 if token.head.i != token.i else 0

    return "\t".join([
        str(idx),                    # ID
        token.text,                   # FORM
        token.lemma_,                 # LEMMA
        token.pos_,                   # UPOS
        token.tag_,                   # XPOS
        "_",                          # FEATS (blank)
        str(head_idx),                # HEAD
        token.dep_,                   # DEPREL
        "_",                          # DEPS (blank)
        "_"                           # MISC (blank)
    ])


def parse_to_conllu(text, nlp):
    """
    Parse text and convert to CoNLL-U format.

    Args:
        text: Input text (tweet)
        nlp: spaCy language model

    Returns:
        CoNLL-U formatted string
    """
    try:
        doc = nlp(text)

        # Process each sentence separately
        conllu_lines = []
        for sent in doc.sents:
            # Add sentence text as comment
            conllu_lines.append(f"# text = {sent.text}")

            # Add each token
            for idx, token in enumerate(sent, start=1):
                conllu_lines.append(token_to_conllu(token, idx))

            # Add blank line between sentences
            conllu_lines.append("")

        return "\n".join(conllu_lines).strip()

    except Exception as e:
        print(f"Error parsing text: {text[:50]}... - {e}", file=sys.stderr)
        return ""


def preprocess_dataset(input_path, output_path, sample_size=None):
    """
    Preprocess the COVIDSenti dataset with syntactic parsing.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        sample_size: Number of samples to process (None for all)
    """
    print(f"Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Reading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original dataset size: {len(df)} tweets")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        print(f"Sampling {sample_size} tweets...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Add column for CoNLL-U parse
    df['conllu_parse'] = ""

    print(f"Parsing {len(df)} tweets...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing tweets"):
        tweet_text = str(row['tweet'])
        conllu = parse_to_conllu(tweet_text, nlp)
        df.at[idx, 'conllu_parse'] = conllu

    # Remove tweets that failed to parse
    df_parsed = df[df['conllu_parse'] != ""].copy()
    print(f"\nSuccessfully parsed: {len(df_parsed)}/{len(df)} tweets")

    if len(df_parsed) < len(df):
        print(f"Failed to parse: {len(df) - len(df_parsed)} tweets")

    # Save results
    print(f"Saving results to {output_path}...")
    df_parsed.to_csv(output_path, index=False)

    print(f"\n✓ Done! Processed dataset saved to {output_path}")
    print(f"  Total tweets: {len(df_parsed)}")
    print(f"  Columns: {df_parsed.columns.tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess COVIDSenti dataset with syntactic parsing")
    parser.add_argument("--input", type=str, default="COVIDSenti.csv",
                        help="Input CSV file path")
    parser.add_argument("--output", type=str, default="COVIDSenti_parsed.csv",
                        help="Output CSV file path")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of samples to process (default: all)")

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.sample)


if __name__ == "__main__":
    main()
