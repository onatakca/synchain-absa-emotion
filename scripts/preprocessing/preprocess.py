import os
import sys
import re
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

def _heuristic_classify(tweet: str) -> str:
   if not isinstance(tweet, str) or not tweet.strip():
      return "not news"
   t = tweet.strip()
   tl = t.lower()
   # Fast URL detection
   if re.search(r"https?://", tl):
      # Likely news if common outlets or headline-y phrasing
      news_domains = [
         "reuters", "apnews", "associated press", "bbc", "cnn", "nytimes",
         "washingtonpost", "wsj", "bloomberg", "guardian", "aljazeera",
         "foxnews", "nbcnews", "abcnews", "cbsnews", "usatoday", "politico",
      ]
      if any(nd in tl for nd in news_domains):
         return "news"
   keywords = [
      "breaking:", "breaking -", "breaking ", "live updates", "live: ",
      "reports", "report:", "according to", "says ", "say ", "announces",
      "confirmed", "official", "press release", "update:", "updates:",
   ]
   if any(k in tl for k in keywords):
      return "news"
   # Headline style: many Title Case words with minimal pronouns
   words = re.findall(r"[A-Za-z]+", t)
   title_like = sum(1 for w in words[:12] if len(w) > 2 and w[0].isupper())
   if title_like >= 6 and not any(p in tl for p in [" i ", " my ", " we ", " me "]):
      return "news"
   return "not news"

def llm_news_categorization(input_file: str, tweet_column: str, output_file: str):
   print("Loading Qwen model for news categorization...")
   # Prefer Qwen2.5-7B-Instruct; allow local path via env QWEN_LOCAL_PATH
   default_model_id = "Qwen/Qwen2.5-7B-Instruct"
   local_candidates = [
      os.environ.get("QWEN_LOCAL_PATH", "").strip(),
      "/home/s2457997/synchain-absa-emotion/models/Qwen2.5-7B-Instruct",
      "/home/s2457997/synchain-absa-emotion/models/Qwen1.5-7B-Chat",
      "/home/s3758869/synchain-absa-emotion/models/Qwen2.5-72B-Instruct",
      "/home/s3758869/synchain-absa-emotion/models/Qwen2.5-7B-Instruct",
   ]
   local_candidates = [p for p in local_candidates if p]

   tokenizer = None
   model = None
   load_errors = []

   # Try local paths first to avoid network issues on cluster
   for path in local_candidates:
      if os.path.isdir(path):
         try:
            print(f"Trying local model at: {path}")
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
               path,
               local_files_only=True,
               trust_remote_code=True,
               torch_dtype="auto",
               device_map="auto",
            )
            break
         except Exception as e:
            load_errors.append(f"Local load failed at {path}: {e}")

   # If no local model loaded, try remote unless offline mode is enforced
   offline = os.environ.get("HF_HUB_OFFLINE", "").strip() or os.environ.get("TRANSFORMERS_OFFLINE", "").strip()
   if model is None and not offline:
      try:
         print(f"Falling back to remote model: {default_model_id}")
         tokenizer = AutoTokenizer.from_pretrained(default_model_id, trust_remote_code=True)
         model = AutoModelForCausalLM.from_pretrained(
            default_model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
         )
      except Exception as e:
         print("Could not load model from Hugging Face; will use heuristic fallback.")
         for err in load_errors:
            print(err)
         print(f"Remote load error: {e}")
         print("Tip: set QWEN_LOCAL_PATH to a local model directory (e.g., models/Qwen2.5-7B-Instruct) to use LLM.")
         model = None
         tokenizer = None
   elif model is None and offline:
      print("HF offline mode is set and no local model path found; using heuristic fallback.")

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

   examples = _get_5_shot_examples() if model is not None else None
   tweets = df[tweet_column].tolist()
   Path(output_file).parent.mkdir(parents=True, exist_ok=True)

   # Process with incremental saving and graceful interrupt handling
   mode_desc = "LLM" if model is not None else "heuristic"
   print(f"Classifying tweets with incremental saving (mode={mode_desc})...")
   save_every = 5
   try:
      iterator = enumerate(tweets)
      if _HAS_TQDM:
         iterator = enumerate(tqdm(tweets, desc="News classification"))
      for idx, t in iterator:
         # Skip already classified
         if "news_category" in df.columns and isinstance(df.at[idx, "news_category"], str):
            continue
         if model is not None:
            label = _classify_tweet(model, tokenizer, t, examples)
         else:
            label = _heuristic_classify(t)
         df.at[idx, "news_category"] = label
         # Print immediate feedback for logs
         preview = str(t).replace("\n", " ").strip()
         if len(preview) > 240:
            preview = preview[:240] + "..."
         print(f"[{idx+1}/{len(tweets)}] {label}: {preview}")
         sys.stdout.flush()
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
