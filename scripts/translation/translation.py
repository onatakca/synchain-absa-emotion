import os
import re
import json
from typing import Dict, List
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig

INPUT_XLSX = "/home/s3758869/synchain-absa-emotion/data/input_data/minsi_data/chinese/sampled_label_comments.xlsx"
OUTPUT_CSV = "/home/s3758869/synchain-absa-emotion/data/input_data/minsi_data/english_translation/sampled_label_comments_translated.csv"
CACHE_JSONL = OUTPUT_CSV + ".cache.jsonl"

MODEL_ID = "/home/s3758869/models/Yi-34B-Chat"

BATCH_SIZE = 4
MAX_NEW_TOKENS_PASS1 = 256
MAX_NEW_TOKENS_PASS2 = 256
TEMPERATURE = 0.0
TOP_P = 1.0
USE_4BIT = True


ZW_RE   = re.compile(r"[\u200b\ufeff]")  
URL_RE  = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
TAG_RE  = re.compile(r"#([^#\s]{1,60})#?|#(\w{1,60})")

def preprocess(text: str):
   text = "" if text is None else str(text)

   text = ZW_RE.sub("", text)
   text = URL_RE.sub("[URL]", text)
   text = USER_RE.sub("[USER]", text)

   tags = []
   for m in TAG_RE.finditer(text):
      t = (m.group(1) or m.group(2) or "").strip()
      if t and t not in tags:
         tags.append(t)

   content = TAG_RE.sub(" ", text)
   content = " ".join(content.split()).strip()
   zh_text = " ".join(text.split()).strip()

   return content, tags, zh_text

def preprocess_tweet(raw: str) -> Dict:
   content, tags, zh_text = preprocess(raw)
   return {"zh_text": zh_text, "content": content, "tags": tags}


def load_model_and_tokenizer():
   tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, trust_remote_code=True)

   if USE_4BIT:
      bnb_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_compute_dtype=torch.float16,
         bnb_4bit_use_double_quant=True,
         bnb_4bit_quant_type="nf4",
      )
      model = AutoModelForCausalLM.from_pretrained(
         MODEL_ID,
         device_map="auto",
         trust_remote_code=True,
         quantization_config=bnb_config,
      )
   else:
      model = AutoModelForCausalLM.from_pretrained(
         MODEL_ID,
         device_map="auto",
         trust_remote_code=True,
         torch_dtype="auto",
      )
   return model, tok

def chat_prompt_pass1(item):
   sys = (
      "Translate the Chinese text to English.\n"
      "Return ONLY a valid JSON object with exactly one key: translation.\n"
      "The value must be English. Do NOT output Chinese.\n"
      "Do NOT include the word translation outside the JSON.\n"
      "Example: {\"translation\":\"...\"}"
   )
   return [
      {"role": "system", "content": sys},
      {"role": "user", "content": item["content"]},
   ]


def looks_untranslated(en_text: str) -> bool:
   t = (en_text or "").strip()
   if not t:
      return True
   if CJK_RE.search(t):
      return True
   return False
 
def chat_prompt_pass2(item: Dict, draft_en: str) -> List[Dict[str, str]]:
   sys = "Improve the following English translation for fluency, but output only the improved translation, nothing else."
   prompt = draft_en
   return [
      {"role": "system", "content": sys},
      {"role": "user", "content": prompt},
   ]

def extract_first_json_object(s: str) -> Dict:
   s = s.strip()

   start = s.find("{")
   if start == -1: raise ValueError("No JSON object start '{' found in model output")

   decoder = json.JSONDecoder()
   obj, end = decoder.raw_decode(s[start:]) 
   return obj

def generate_text(model, tok, messages, max_new_tokens: int) -> str:
   if hasattr(tok, "apply_chat_template"):
      prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   else:
      prompt = ""
      for m in messages:
         prompt += f"{m['role'].upper()}: {m['content']}\n"
      prompt += "ASSISTANT: "

   inputs = tok(prompt, return_tensors="pt").to(model.device)

   im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

   eos_ids = [im_end_id] if im_end_id is not None and im_end_id != tok.unk_token_id else [tok.eos_token_id]

   with torch.no_grad():
      out = model.generate(
         **inputs,
         max_new_tokens=max_new_tokens,
         do_sample=False,
         eos_token_id=eos_ids,
         pad_token_id=tok.eos_token_id,
      )

   text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
   return text


def load_cache(cache_path: str) -> Dict[int, Dict]:
   cache: Dict[int, Dict] = {}
   if not os.path.exists(cache_path):
      return cache
   with open(cache_path, "r", encoding="utf-8") as f:
      for line in f:
         line = line.strip()
         if not line:
               continue
         obj = json.loads(line)
         cache[int(obj["tweet_id"])] = obj
   return cache

def append_cache(cache_path: str, obj: Dict) -> None:
   with open(cache_path, "a", encoding="utf-8") as f:
      f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def is_usable_content(content: str) -> bool:
   if content is None:
      return False
   content = content.strip()
   if not content:
      return False
   if re.fullmatch(r"[\d\W_]+", content):
      return False
   if len(content) < 2:  
      return False
   return True


CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def normalize_translation(raw_out: str) -> str:
   s = (raw_out or "").strip()

   try:
      obj = extract_first_json_object(s)
      if isinstance(obj, dict) and "translation" in obj:
         t = str(obj["translation"]).strip()
         return t
   except Exception:
      pass
   m = re.search(r'translation\s*["\']?\s*[:=]\s*["\']?(.*?)["\']?\s*$', s, flags=re.I | re.S)
   if m:
      t = m.group(1).strip()
      t = t.strip().strip('"}').strip()
      return t

   s = re.split(r"\n\s*\n", s, maxsplit=1)[0].strip()
   return s


def write_csv_from_cache(cache_path: str, output_csv: str):
   rows = []
   with open(cache_path, "r", encoding="utf-8") as f:
      for line in f:
         line = line.strip()
         if line:
               rows.append(json.loads(line))
   if not rows:
      return
   out_df = pd.DataFrame(rows).sort_values("tweet_id")
   out_df = out_df[["tweet_id", "zh_text", "en_text", "label"]]
   out_df.to_csv(output_csv, index=False, encoding="utf-8")

def run():
   df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
   if "comments" not in df.columns or "label" not in df.columns:
      raise ValueError("Expected Excel columns: comments, label")

   df = df.reset_index(drop=True)
   df["tweet_id"] = df.index.astype(int)

   cache = load_cache(CACHE_JSONL)

   model, tok = load_model_and_tokenizer()

   results: List[Dict] = []
   n = len(df)

   for start in tqdm(range(0, n, BATCH_SIZE), desc="Batches", unit="batch"):
      batch = df.iloc[start:start + BATCH_SIZE]

      for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Translating rows {start}-{min(start+BATCH_SIZE-1, n-1)}", unit="tweet", leave=False):
         tid = int(row["tweet_id"])
         if tid in cache:
            results.append(cache[tid])
            continue

         raw = row["comments"]
         label = row["label"]

         item = preprocess_tweet(raw)

         if not is_usable_content(item["content"]):
            out_obj = {
               "tweet_id": tid,
               "zh_text": item["zh_text"],
               "en_text": "",
               "label": label,
            }
            append_cache(CACHE_JSONL, out_obj)
            cache[tid] = out_obj
            results.append(out_obj)
            continue

         raw_out = generate_text(model, tok, chat_prompt_pass1(item), MAX_NEW_TOKENS_PASS1)
         en_text = normalize_translation(raw_out)
         
         if looks_untranslated(en_text):
            raw_out2 = generate_text(model, tok, chat_prompt_pass1(item), MAX_NEW_TOKENS_PASS1)
            en_text2 = normalize_translation(raw_out2)
            if not looks_untranslated(en_text2):
               en_text = en_text2

         out_obj = {
            "tweet_id": tid,
            "zh_text": item["zh_text"],   # Chinese 
            "en_text": en_text, # English translation 
            "label": label,
         }

         append_cache(CACHE_JSONL, out_obj)
         cache[tid] = out_obj
         results.append(out_obj)

         if (start // BATCH_SIZE) % 50 == 0:
            write_csv_from_cache(CACHE_JSONL, OUTPUT_CSV)

   out_df = pd.DataFrame(results).sort_values("tweet_id")
   out_df = out_df[["tweet_id", "zh_text", "en_text", "label"]]
   out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

   print(f"Wrote: {OUTPUT_CSV}")
   print(f"Cache: {CACHE_JSONL}")

if __name__ == "__main__":
   run()
