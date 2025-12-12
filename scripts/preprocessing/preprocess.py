import pandas as pd
from ftfy import fix_text
import emoji

INPUT_FILE = "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/raw/COVIDSenti-Full.csv"
OUTPUT_FILE = "/home/s3758869/synchain-absa-emotion/data/input_data/COVIDSenti/processed/COVIDSenti_full_proc.csv"

def fix_and_remove_emojis(s: str) -> str:
   s = fix_text(str(s))               
   s = emoji.replace_emoji(s, "")      
   return s

df = pd.read_csv(INPUT_FILE)

for col in df.columns:
   if df[col].dtype == object:
      df[col] = df[col].apply(fix_and_remove_emojis)

df.to_csv(OUTPUT_FILE, index=False)
