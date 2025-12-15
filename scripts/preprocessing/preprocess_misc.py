from pathlib import Path
import re
from pathlib import Path
from ftfy import fix_text
import emoji
import pandas as pd

dataset_specs = [
      {
         "name": "SenWave",
         "path": Path("SenWave/processed/senwave_not_news_proc.csv"),
         "tweet_column": "tweet",
      },
      {
         "name": "COVIDSenti_full",
         "path": Path("COVIDSenti/processed/covidsenti_full_not_news_proc.csv"),
         "tweet_column": "tweet",
      },
      {
         "name": "COVID_19_NLP_test",
         "path": Path("COVID-19-NLP/processed/corona_nlp_test_not_news_proc.csv"),
         "tweet_column": "tweet",
      },
      {
         "name": "COVID_19_NLP_train",
         "path": Path("COVID-19-NLP/processed/corona_nlp_train_not_news_proc.csv"),
         "tweet_column": "tweet",
      },
   ]

SENWAVE_DROP_COLUMNS = ["Denial","Official report","Joking"]

def fix_and_remove_emojis(s: str) -> str:
   s = fix_text(str(s))               
   s = emoji.replace_emoji(s, "")      
   return s

URL_PATTERN = re.compile(r"http\S+|www\.\S+")

def remove_links(tweet: str) -> str:
   return URL_PATTERN.sub("", tweet)
 
def remove_space_tweet(tweet: str) -> str:
   tweet = tweet.replace("\n", " ").replace("\r", " ")
   tweet = re.sub(r"\s+", " ", tweet)
   return tweet.strip()
 
 
def preprocess_tweet(tweet: str) -> str:
   tweet = fix_and_remove_emojis(tweet) 
   tweet = remove_links(tweet)
   tweet = remove_space_tweet(tweet)
   return tweet

for dataset in dataset_specs:
   print(f"Processing {dataset['name']}...")
   data = pd.read_csv(dataset["path"])
   if dataset["name"] == "SenWave":
      initial_rows = len(data)
      for col in SENWAVE_DROP_COLUMNS:
         if col in data.columns:
            data = data[data[col] != 1]
      removed_rows = initial_rows - len(data)
      print(f"Removed {removed_rows} rows with annotations in {SENWAVE_DROP_COLUMNS}")
      data = data.drop(columns=SENWAVE_DROP_COLUMNS, errors='ignore')
   data["tweet"] = data["tweet"].astype(str).apply(preprocess_tweet)
   data.to_csv(dataset["path"], index=False)
   print(f"Saved {len(data)} rows to {dataset['path']}")