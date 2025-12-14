import pandas as pd
from ftfy import fix_text
import emoji
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
   from tqdm.auto import tqdm
   _HAS_TQDM = True
except Exception:
   _HAS_TQDM = False

# to run:
# /home/s2457997/.venv/bin/python /home/s2457997/synchain-absa-emotion/scripts/preprocessing/preprocess.py
BASE = "/home/s2457997/synchain-absa-emotion/data/input_data"
# BASE = "/home/s3758869/synchain-absa-emotion/data/input_data"
# BASE = "/home/s0000000/synchain-absa-emotion/data/input_data"


def fix_and_remove_emojis(s: str) -> str:
   s = fix_text(str(s))               
   s = emoji.replace_emoji(s, "")      
   return s

def preprocess_dataset(input_file, output_file, tweet_column):
   df = pd.read_csv(input_file, encoding="latin1")

   if tweet_column in df.columns:
      df[tweet_column] = df[tweet_column].apply(fix_and_remove_emojis)

   Path(output_file).parent.mkdir(parents=True, exist_ok=True)
   df.to_csv(output_file, index=False)

# news categorization with Qwen 7B

def _get_5_shot_examples() -> str:
   return (
      """
      Here are some examples of tweets. Classify them as 'news' or 'not news'.

      Tweet: "US stock futures and Asian shares fall after Trump says he and first lady have tested positive for Covid-19 https://t.co/y2a5Z3n5qS https://t.co/n534Mcm2cT"
      Classification: news

      Tweet: "Oil prices fall as much as 5% after Trump tests positive for COVID-19 https://t.co/38pLd2i37l https://t.co/9r4V3A0p3x"
      Classification: news

      Tweet: "Trump says he and first lady have tested positive for coronavirus - follow live https://t.co/d5R43O9033"
      Classification: news

      Tweet: "President Trump and the first lady have tested positive for Covid-19. The announcement comes after a top aide, Hope Hicks, tested positive for the virus. Here's what we know. https://t.co/oW23SjI95f"
      Classification: news

      Tweet: "BREAKING: President Trump and first lady Melania Trump test positive for Covid-19 https://t.co/w5nU8p2d5X"
      Classification: news
      """
   )

def _build_prompt(tweet: str, examples: str) -> str:
   return f"""{examples}
      Tweet: "{tweet}"
      Classification:"""

def _classify_tweet(model, tokenizer, tweet: str, prompt_template: str) -> str:
   if not isinstance(tweet, str) or not tweet.strip():
      return "not news"

   prompt = _build_prompt(tweet, prompt_template)
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = model.generate(
      **inputs,
      max_new_tokens=5,
      pad_token_id=tokenizer.eos_token_id,
   )
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   generated_text = response[len(prompt):].strip().lower()
   if "news" in generated_text:
      return "news"
   return "not news"

def llm_news_categorization(input_file: str, tweet_column: str, output_file: str):
   print("Loading Qwen model for news categorization...")
   model_name = "Qwen/Qwen1.5-7B-Chat"

   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype="auto",
      device_map="auto",
   )

   print(f"Reading preprocessed data: {input_file}")
   df = pd.read_csv(input_file, encoding="latin1")
   if tweet_column not in df.columns:
      raise ValueError(f"Column '{tweet_column}' not found in {input_file}")

   # Polite resource usage: cap CPU threads to be nice on shared systems
   try:
      torch.set_num_threads(max(1, torch.get_num_threads() // 2))
   except Exception:
      pass

   # Resume support: if output exists, load and keep previously classified rows
   existing = None
   if Path(output_file).exists():
      try:
         existing = pd.read_csv(output_file, encoding="latin1")
      except Exception:
         existing = None
   if existing is not None and "news_category" in existing.columns:
      # Align on index length; merge by position to keep simple and robust
      if len(existing) == len(df):
         df["news_category"] = existing["news_category"]
      else:
         # Fallback: only copy what's available
         available = min(len(existing), len(df))
         df.loc[:available-1, "news_category"] = existing.loc[:available-1, "news_category"]

   examples = _get_5_shot_examples()
   tweets = df[tweet_column].tolist()
   Path(output_file).parent.mkdir(parents=True, exist_ok=True)

   # Process with incremental saving and graceful interrupt handling
   print("Classifying tweets with incremental saving...")
   save_every = 100
   try:
      iterator = enumerate(tweets)
      if _HAS_TQDM:
         iterator = enumerate(tqdm(tweets, desc="News classification"))
      for idx, t in iterator:
         # Skip already classified
         if "news_category" in df.columns and isinstance(df.at[idx, "news_category"], str):
            continue
         df.at[idx, "news_category"] = _classify_tweet(model, tokenizer, t, examples)
         if (idx + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
      # Final save
      df.to_csv(output_file, index=False)
   except KeyboardInterrupt:
      print("Interrupted. Saving partial results...")
      df.to_csv(output_file, index=False)
   except Exception as e:
      print(f"Error during classification: {e}. Saving partial results...")
      df.to_csv(output_file, index=False)

# TODO : Salih
# note : from SenWave remove all that are in category Denial,Official report,Joking
# note : from covid_sentti remove labels and from COVID-19-NLP all other columns
# output files for COVID-19-NLP and CovidSneti must be tweet, original_sentiment

# todo : Salih
# from all non news tweets, remove
# links

def llm_tweet_annotation(model, tokenizer, input_file, output_file, tweet_column):
   #TODO: Salih to implement LLM news annotation. 
   # remove also non informational tweets like : #media #corona vrius 
   # or like #coronasucks .
   # add column is_news or is_usefu
   pass

def main():
   datasets = [
      {
         "input": f"{BASE}/SenWave/raw/SenWave.csv",
         "output": f"{BASE}/SenWave/processed/SenWave_proc.csv",
         "tweet_column": "Tweet",
      },
      {
         "input": f"{BASE}/COVIDSenti/raw/COVIDSenti-Full.csv",
         "output": f"{BASE}/COVIDSenti/processed/COVIDSenti_full_proc.csv",
         "tweet_column": "tweet",
      },
      {
         "input": f"{BASE}/COVID-19-NLP/raw/Corona_NLP_test.csv",
         "output": f"{BASE}/COVID-19-NLP/processed/Corona_NLP_test_proc.csv",
         "tweet_column": "OriginalTweet",
      },
      {
         "input": f"{BASE}/COVID-19-NLP/raw/Corona_NLP_train.csv",
         "output": f"{BASE}/COVID-19-NLP/processed/Corona_NLP_train_proc.csv",
         "tweet_column": "OriginalTweet",
      },
   ]
   for dataset in datasets:
      preprocess_dataset(dataset["input"], dataset["output"], dataset["tweet_column"])

   # Run news categorization only for COVID-19-NLP test split to validate pipeline
   llm_news_categorization(
      input_file=f"{BASE}/COVID-19-NLP/processed/Corona_NLP_test_proc.csv",
      tweet_column="OriginalTweet",
      output_file="/home/s2457997/synchain-absa-emotion/data/output_data/Corona_NLP_test_categorized.csv",
   )

if __name__ == "__main__":
   main()
