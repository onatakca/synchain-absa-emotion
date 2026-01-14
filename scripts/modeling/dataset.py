import json
from torch.utils.data import Dataset
import torch

from scripts.qwen_model.prompts import (
   prompt_aspect_extraction_no_res,
   prompt_emotion_classification,
   prompt_opinion_extraction,
   prompt_sentiment_classification,
   prompt_syntactic_parsing,
)

def get_data(json_ann_files):
   data = {}
   for file in json_ann_files:
      with open(file, "r") as f:
         _data = json.load(f)
      data.update(_data)
   return data


class ABSADataset(Dataset):
   def __init__(self, data, tasks, max_len=512, max_samples=None):
      self.data = data
      self.examples = []
      self.tasks = tasks
      self.max_samples = max_samples

      for tweet, rec in self.data.items():
         aspects_dict = rec["aspects"]
         aspect_terms = list(aspects_dict.keys())

         conllu_parse = rec["conllu_parse"]
         aspect_sentiments_raw = rec["aspect_sentiments_raw"]       
         aspect_sentiments_label = rec["aspect_sentiments_label"]   
         aspect_syntactic = rec["aspect_syntactic"]                 
         aspect_opinions = rec["aspect_opinions"]                   
         aspect_emotions_raw = rec["aspect_emotions_raw"]           

         prompts = prompt_aspect_extraction_no_res([tweet], [conllu_parse]) 
         target_text = " ".join(aspect_terms)  
         self.examples.append({
               "task": "aspect_extraction",
               "prompt_text": prompts[0],
               "target_text": target_text,
         })

         for j, aspect in enumerate(aspect_terms):
               k = str(j)

               if "aspect_syntactic_parsing" in self.tasks:
                  prompts = prompt_syntactic_parsing([tweet], [aspect], [conllu_parse])
                  self.examples.append({
                     "task": "aspect_syntactic_parsing",
                     "prompt_text": prompts[0],
                     "target_text": aspect_syntactic[k],
                  })

               if "aspect_opinion_extraction" in self.tasks:
                  prompts = prompt_opinion_extraction([tweet], [aspect], [aspect_syntactic[k]])
                  self.examples.append({
                     "task": "aspect_opinion_extraction",
                     "prompt_text": prompts[0],
                     "target_text": aspect_opinions[k],
                  })

               if "aspect_sentiment_analysis" in self.tasks:
                  prompts = prompt_sentiment_classification([tweet], [aspect], [aspect_opinions[k]])
                  self.examples.append({
                     "task": "aspect_sentiment_analysis",
                     "prompt_text": prompts[0],
                     "target_text": aspect_sentiments_raw[k],
                  })

               if "aspect_emotion_detection" in self.tasks:
                  prompts = prompt_emotion_classification(
                     [tweet],
                     [aspect],
                     [aspect_opinions[k]],
                     [aspect_sentiments_label[k]],
                  )
                  self.examples.append({
                     "task": "aspect_emotion_detection",
                     "prompt_text": prompts[0],
                     "target_text": aspect_emotions_raw[k],
                  })
      if max_samples is not None:
         self.examples = self.examples[:max_samples]
         
   def __getitem__(self, idx):
      return self.examples[idx]

   def __len__(self):
      return len(self.examples)

def tokenize(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text
 

class Collator:
   def __init__(self, tokenizer, max_length):
      self.tokenizer = tokenizer
      self.max_length = max_length
      if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token

   def __call__(self, batch):
      prompt_strs = [ tokenize(self.tokenizer, item["prompt_text"]) for item in batch ]

      full_strs = [
         p + item["target_text"] + self.tokenizer.eos_token for p, item in zip(prompt_strs, batch)
      ]

      full_toks = self.tokenizer(
         full_strs,
         padding=True,
         truncation=True,
         max_length=self.max_length,
         add_special_tokens=False,
         return_tensors="pt",
      )

      prompt_toks = self.tokenizer(
         prompt_strs,
         padding=False,
         truncation=True,
         max_length=self.max_length,
         add_special_tokens=False,
         return_tensors=None,
      )
      prompt_lens = [len(ids) for ids in prompt_toks["input_ids"]]

      labels = full_toks["input_ids"].clone()
      for i, plen in enumerate(prompt_lens):
         labels[i, :plen] = -100

      return {
         "input_ids": full_toks["input_ids"],
         "attention_mask": full_toks["attention_mask"],
         "labels": labels,
      }