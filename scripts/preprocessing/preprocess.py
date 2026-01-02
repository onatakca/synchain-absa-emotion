import os
import re
import sys
from pathlib import Path

import emoji
import pandas as pd
import torch
from ftfy import fix_text
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path("/home/s3758869/synchain-absa-emotion")
DATA_DIR = PROJECT_ROOT / "data" / "input_data"


def fix_and_remove_emojis(s: str) -> str:
    s = fix_text(str(s))
    s = emoji.replace_emoji(s, "")
    return s


def is_tweet_cut_off(tweet, pattern="‚Ä¶"):
    if not isinstance(tweet, str):
        return False
    return tweet.endswith(pattern)


def strip_urls(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def standardize_sentiment(sentiment: str) -> str:
    if not isinstance(sentiment, str):
        return sentiment
    sentiment_lower = sentiment.lower().strip()
    sentiment_map = {"neu": "neutral", "pos": "positive", "neg": "negative"}
    return sentiment_map.get(sentiment_lower, sentiment)


def _get_5_shot_examples() -> str:
    return """Here are some examples of tweets. Classify them as 'news' or 'not news'.

Tweet: "US stock futures and Asian shares fall after Trump says he and first lady have tested positive for Covid-19 https://t.co/y2a5Z3n5qS https://t.co/n534Mcm2cT"
Classification: news

Tweet: "Oil prices fall as much as 5% after Trump tests positive for COVID-19 https://t.co/38pLd2i37l https://t.co/9r4V3A0p3x"
Classification: news

Tweet: "Trump says he and first lady have tested positive for coronavirus - follow live https://t.co/d5R43O9033"
Classification: news

Tweet: "President Trump and the first lady have tested positive for Covid-19. The announcement comes after a top aide, Hope Hicks, tested positive for the virus. Here's what we know. https://t.co/oW23SjI95f"
Classification: news

Tweet: "Panic food buying in Germany due to #coronavirus has begun.  But the #organic is left behind! #Hamsterkauf"
Classification: not news

Tweet: "#Covid_19 Went to the Grocery Store, turns out all cleaning supplies have been bought out for fear of Coronavirus."
Classification: not news

Tweet: "Virologists weigh in on novel coronavirus in China's outbreak https://t.co/w5nU8p2vf3"
Classification: news

Tweet: "Abu Dhabi launches coronavirus website"
Classification: news

Tweet: First Coronavirus death outside of China as man, 44, dies in Philippines
Classification: news

Tweet: Harvard and MIT tell students not to return from spring break due to coronavirus - MIT Technology Review
Classification: news

Tweet: "BREAKING: President Trump and first lady Melania Trump test positive for Covid-19 https://t.co/w5nU8p2d5X"
Classification: news"""


def _build_prompt(tweet: str, examples: str) -> str:
    return f"""{examples}
Tweet: "{tweet}"
Classification:"""


def _classify_tweet(model, tokenizer, tweet: str, prompt_template: str) -> str:
    if not isinstance(tweet, str) or not tweet.strip():
        return "not news"

    prompt = _build_prompt(tweet, prompt_template)
    inputs = tokenizer(prompt, return_tensors="pt")

    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(prompt) :].strip().lower()
    if "news" in generated_text:
        return "news"
    return "not news"


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return sanitized.strip("_").lower()


def load_news_classifier():
    model_path = "/home/s3758869/synchain-absa-emotion/models/Qwen2.5-7B-Instruct"

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"Model loaded successfully")

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))

    return model, tokenizer


def build_dataset_configurations():
    dataset_specs = [
        {
            "name": "SenWave",
            "relative_input": Path("SenWave/raw/SenWave.csv"),
            "tweet_column": "Tweet",
        },
        {
            "name": "COVIDSenti_full",
            "relative_input": Path("COVIDSenti/raw/COVIDSenti-Full.csv"),
            "tweet_column": "tweet",
        },
        {
            "name": "COVID_19_NLP_test",
            "relative_input": Path("COVID-19-NLP/raw/Corona_NLP_test.csv"),
            "tweet_column": "OriginalTweet",
        },
        {
            "name": "COVID_19_NLP_train",
            "relative_input": Path("COVID-19-NLP/raw/Corona_NLP_train.csv"),
            "tweet_column": "OriginalTweet",
        },
    ]

    configurations = []
    for spec in dataset_specs:
        dataset_root = spec["relative_input"].parts[0]
        dataset_dir = DATA_DIR / dataset_root

        input_path = DATA_DIR / spec["relative_input"]
        processed_dir = dataset_dir / "processed"
        sanitized_stem = _sanitize_name(spec["relative_input"].stem)
        processed_path = processed_dir / f"{sanitized_stem}_proc.csv"

        configurations.append(
            {
                "name": spec["name"],
                "tweet_column": spec["tweet_column"],
                "input_path": input_path,
                "processed_path": processed_path,
                "processed_dir": processed_dir,
                "sanitized_stem": sanitized_stem,
            }
        )

    return configurations


def llm_tweet_annotation(
    model, tokenizer, input_file, output_file, tweet_column, save_frequency=25
):
    print(f"Annotating {input_file} → {output_file}")
    dataframe = pd.read_csv(input_file, encoding="latin1")

    if tweet_column not in dataframe.columns:
        raise ValueError(f"Column '{tweet_column}' not found in {input_file}")

    existing_annotations = None
    if Path(output_file).exists():
        try:
            existing_annotations = pd.read_csv(output_file, encoding="latin1")
        except Exception:
            existing_annotations = None

    if (
        existing_annotations is not None
        and "news_category" in existing_annotations.columns
    ):
        if len(existing_annotations) == len(dataframe):
            dataframe["news_category"] = existing_annotations["news_category"]
        else:
            overlap_count = min(len(existing_annotations), len(dataframe))
            dataframe.loc[: overlap_count - 1, "news_category"] = (
                existing_annotations.loc[: overlap_count - 1, "news_category"]
            )

    examples = _get_5_shot_examples()
    tweets = dataframe[tweet_column].tolist()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"Classifying tweets with incremental saving...")

    try:
        for index, tweet_text in enumerate(tqdm(tweets, desc="News classification")):
            already_labeled = "news_category" in dataframe.columns and isinstance(
                dataframe.at[index, "news_category"], str
            )
            if already_labeled:
                continue

            if is_tweet_cut_off(tweet_text):
                label = "news"
            else:
                label = _classify_tweet(model, tokenizer, tweet_text, examples)
            dataframe.at[index, "news_category"] = label

            preview_text = str(tweet_text).replace("\n", " ").strip()
            if len(preview_text) > 240:
                preview_text = preview_text[:240] + "..."
            print(f"[{index + 1}/{len(tweets)}] {label}: {preview_text}")
            sys.stdout.flush()

            if (index + 1) % save_frequency == 0:
                dataframe.to_csv(output_file, index=False)

        dataframe.to_csv(output_file, index=False)
    except KeyboardInterrupt:
        print("Interrupted. Saving partial results...")
        dataframe.to_csv(output_file, index=False)
    except Exception as error:
        print(f"Error during classification: {error}. Saving partial results...")
        dataframe.to_csv(output_file, index=False)


def main():
    datasets = build_dataset_configurations()

    def load_and_transform_dataset(cfg):
        df = pd.read_csv(cfg["input_path"], encoding="latin1")
        name = cfg["name"].lower()
        if name == "senwave":
            if "Tweet" not in df.columns:
                raise ValueError("SenWave: 'Tweet' column missing")
            for col in ["Denial", "Official report", "Joking"]:
                if col not in df.columns:
                    raise ValueError(f"SenWave: expected column '{col}' not found")
            mask_drop = (
                (df["Denial"] == 1) | (df["Official report"] == 1) | (df["Joking"] == 1)
            )
            df = df.loc[~mask_drop].copy()
            df.rename(columns={"Tweet": "tweet"}, inplace=True)
            df["tweet"] = df["tweet"].apply(fix_and_remove_emojis)
            return df
        elif name == "covidsenti_full":
            needed = ["tweet", "label"]
            for n in needed:
                if n not in df.columns:
                    raise ValueError(f"COVIDSenti: expected column '{n}' not found")
            out = df[["tweet", "label"]].copy()
            out.rename(columns={"label": "original_sentiment"}, inplace=True)
            out["tweet"] = out["tweet"].apply(fix_and_remove_emojis)
            out["original_sentiment"] = out["original_sentiment"].apply(
                standardize_sentiment
            )
            return out
        elif name in ("covid_19_nlp_test", "covid_19_nlp_train"):
            needed = ["OriginalTweet", "Sentiment"]
            for n in needed:
                if n not in df.columns:
                    raise ValueError(f"COVID-19-NLP: expected column '{n}' not found")
            out = df[["OriginalTweet", "Sentiment"]].copy()
            out.rename(
                columns={"OriginalTweet": "tweet", "Sentiment": "original_sentiment"},
                inplace=True,
            )
            out["tweet"] = out["tweet"].apply(fix_and_remove_emojis)
            out["original_sentiment"] = out["original_sentiment"].apply(
                standardize_sentiment
            )
            return out
        else:
            raise ValueError(f"Unknown dataset name: {cfg['name']}")

    for cfg in tqdm(datasets, desc="Processing datasets"):
        cfg["processed_dir"].mkdir(parents=True, exist_ok=True)
        df_proc = load_and_transform_dataset(cfg)
        df_proc.to_csv(cfg["processed_path"], index=False)

    model, tokenizer = load_news_classifier()
    for cfg in tqdm(datasets, desc="Annotating datasets"):
        tmp_annot = (
            cfg["processed_dir"] / f"{cfg['sanitized_stem']}_categorised_tmp.csv"
        )
        llm_tweet_annotation(
            model=model,
            tokenizer=tokenizer,
            input_file=cfg["processed_path"],
            output_file=tmp_annot,
            tweet_column="tweet",
        )

        annotated = pd.read_csv(tmp_annot, encoding="latin1")
        if "news_category" not in annotated.columns:
            raise ValueError(
                f"Annotation failed for {cfg['name']}: 'news_category' missing"
            )

        news_df = annotated[annotated["news_category"].str.lower() == "news"].copy()
        not_news_df = annotated[annotated["news_category"].str.lower() != "news"].copy()

        if "tweet" in not_news_df.columns:
            not_news_df["tweet"] = not_news_df["tweet"].apply(strip_urls)

        for d in (news_df, not_news_df):
            if "news_category" in d.columns:
                d.drop(columns=["news_category"], inplace=True)

        final_news = cfg["processed_dir"] / f"{cfg['sanitized_stem']}_news_proc.csv"
        final_not_news = (
            cfg["processed_dir"] / f"{cfg['sanitized_stem']}_not_news_proc.csv"
        )

        news_df.to_csv(final_news, index=False)
        not_news_df.to_csv(final_not_news, index=False)
        try:
            if tmp_annot.exists():
                tmp_annot.unlink()
        except Exception:
            pass
        for p in cfg["processed_dir"].glob(f"{cfg['sanitized_stem']}_*.csv"):
            if p.name not in {final_news.name, final_not_news.name}:
                try:
                    p.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
