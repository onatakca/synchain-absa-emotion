from transformers import TrainerCallback
from scripts.qwen_model.qwen_model import generate_batch
from scripts.annotation.parsing import extract_aspects,extract_sentiment, extract_emotion
import numpy as np
import torch

def accuracy(targets, preds):
   targets =  np.asarray(targets)
   preds =  np.asarray(preds)
   
   return np.sum(targets == preds) / len(targets)
 
class EvaluationCallback(TrainerCallback):
   def __init__(self, val_dataset, tokenizer, max_new_tokens=512, tasks=None, eval_batch_size=8):
      self.val_dataset = val_dataset
      self.tasks = tasks  
      self.tokenizer = tokenizer
      self.max_new_tokens = max_new_tokens
      self.eval_batch_size = eval_batch_size
      
   def on_evaluate(self, args, state, control, model=None, **kwargs):
      print(f"state : {state} evaluation")
      model.eval()
      
      aspect_items = []
      sentiment_items = []
      emotion_items = []
      
      for item in self.val_dataset:
         if self.tasks is not None and item["task"] not in self.tasks:
            continue
         if item["task"] == "aspect_extraction":
            aspect_items.append(item)
         elif item["task"] == "aspect_sentiment_analysis":
            sentiment_items.append(item)
         elif item["task"] == "aspect_emotion_detection":
            emotion_items.append(item)
      
      aspects_tp = 0.0
      aspects_fp = 0.0
      aspects_fn = 0.0
      
      if aspect_items:
         prompts = [item["prompt_text"] for item in aspect_items]
         outputs = generate_batch(model, self.tokenizer, prompts, self.max_new_tokens, batch_size=self.eval_batch_size)
         for item, model_output in zip(aspect_items, outputs):
            teacher_aspects = item["target_text"].split()
            student_aspects = extract_aspects(model_output)
            S, T = set(student_aspects), set(teacher_aspects)
            aspects_tp += len(S & T)
            aspects_fp += len(S - T)
            aspects_fn += len(T - S)
      
      teacher_sentiments, student_sentiments = [], []
      if sentiment_items:
         prompts = [item["prompt_text"] for item in sentiment_items]
         outputs = generate_batch(model, self.tokenizer, prompts, self.max_new_tokens, batch_size=self.eval_batch_size)
         for item, model_output in zip(sentiment_items, outputs):
            teacher_sentiments.append(extract_sentiment(item["target_text"]))
            student_sentiments.append(extract_sentiment(model_output))
      
      teacher_emotions, student_emotions = [], []
      if emotion_items:
         prompts = [item["prompt_text"] for item in emotion_items]
         outputs = generate_batch(model, self.tokenizer, prompts, self.max_new_tokens, batch_size=self.eval_batch_size)
         for item, model_output in zip(emotion_items, outputs):
            teacher_emotions.append(extract_emotion(item["target_text"]))
            student_emotions.append(extract_emotion(model_output))

      aspect_recall = aspects_tp / (aspects_tp + aspects_fn) if (aspects_tp + aspects_fn) > 0 else 0.0
      aspect_precision = aspects_tp / (aspects_tp + aspects_fp) if (aspects_tp + aspects_fp) > 0 else 0.0
      aspect_f1 = 2 * aspect_recall * aspect_precision / (1e-07 + aspect_precision + aspect_recall)
      sentiment_accuracy = accuracy(teacher_sentiments, student_sentiments) if teacher_sentiments else 0.0
      emotion_accuracy = accuracy(teacher_emotions, student_emotions) if teacher_emotions else 0.0
      
      metrics = {
         "gen_eval/aspect_precision": aspect_precision,
         "gen_eval/aspect_recall": aspect_recall,
         "gen_eval/aspect_f1": aspect_f1,
         "gen_eval/sentiment_acc": sentiment_accuracy,
         "gen_eval/emotion_acc": emotion_accuracy,
      }
      
      print(f"Results::")
      for k, v in metrics.items():
         print(f"{k}: {v:.4f}")
      
      if kwargs.get("trainer", None) is not None:
         trainer = kwargs.get("trainer", None)
         trainer.log(metrics)

      if torch.cuda.is_available():torch.cuda.empty_cache()
      
      model.train()
      return metrics