import pandas as pd
from ftfy import fix_text
import emoji
from pathlib import Path

def fix_and_remove_emojis(s: str) -> str:
   s = fix_text(str(s))               
   s = emoji.replace_emoji(s, "")      
   return s

def preprocess_dataset(input_file, output_file, tweet_column):
   df = pd.read_csv(input_file)

   if tweet_column in df.columns:
      df[tweet_column] = df[tweet_column].apply(fix_and_remove_emojis)

   Path(output_file).parent.mkdir(parents=True, exist_ok=True)
   df.to_csv(output_file, index=False)

# TODO : Salih
# note : from SenWave remove all that are in category Denial,Official report,Joking
# note : from covid_sentti remove labels and from COVID-19-NLP all other columns
# output files for COVID-19-NLP and CovidSneti must be tweet, original_sentiment

def llm_tweet_annotation(model, tokenizer, input_file, output_file, tweet_column):
   #TODO: Salih to implement LLM news ann
   pass

def main():
   datasets = [
      {
         "input": "/home/s3758869/synchain-absa-emotion/data/input_data/SenWave/raw/SenWave.csv",
         "output": "/home/s3758869/synchain-absa-emotion/data/input_data/SenWave/processed/SenWave_proc.csv",
         "tweet_column": "Tweet",
      },
      {
         "input": "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/raw/COVIDSenti-Full.csv",
         "output": "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/processed/COVIDSenti_full_proc.csv",
         "tweet_column": "tweet",
      },
      {
         "input": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/raw/Corona_NLP_test.csv",
         "output": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/processed/Corona_NLP_test_proc.csv",
         "tweet_column": "OriginalTweet",
      },
      {
         "input": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/raw/Corona_NLP_train.csv",
         "output": "/home/s3758869/synchain-absa-emotion/data/input_data/COVID-19-NLP/processed/Corona_NLP_train_proc.csv",
         "tweet_column": "OriginalTweet",
      },
   ]

   for dataset in datasets:
      preprocess_dataset(dataset["input"], dataset["output"], dataset["tweet_column"])

if __name__ == "__main__":
   main()
