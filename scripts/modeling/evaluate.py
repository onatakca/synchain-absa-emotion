from transformers import TrainerCallback
from scripts.qwen_model.qwen_model import generate_response
from scripts.annotation.parsing import extract_aspects,extract_sentiment, extract_emotion
import numpy as np
import torch

def accuracy(targets, preds):
   targets =  np.asarray(targets)
   preds =  np.asarray(preds)
   
   return np.sum(targets == preds) / len(targets)
 
class EvaluationCallback(TrainerCallback):
   def __init__(self, val_dataset, tokenizer, max_new_tokens=512, tasks=None):
      self.val_dataset = val_dataset
      self.tasks = tasks  
      self.tokenizer = tokenizer
      self.max_new_tokens = max_new_tokens
      
   def on_evaluate(self, args, state, control, model=None, **kwargs):
      print(f"epoch {state.epoch} evaluation")
      model.eval()
      
      aspects_tp = 0.0
      aspects_fp = 0.0
      aspects_fn = 0.0      
      
      teacher_sentiments = []
      student_sentiments = []
      
      teacher_emotions = []
      student_emotions = []
      
      with torch.no_grad():
         for item in self.val_dataset:
            if self.tasks is not None and item["task"] not in self.tasks:
               continue
            if item["task"] == "aspect_extraction":
               teacher_aspects = item["target_text"].split()
               model_output = generate_response(model, self.tokenizer,item["prompt_text"], max_new_tokens=self.max_new_tokens)
               student_aspects = extract_aspects(model_output)
               
               S = set(student_aspects)
               T = set(teacher_aspects)
               
               aspects_tp += len(S & T)  
               aspects_fp += len(S - T)  
               aspects_fn += len(T - S)
               
            elif item["task"] == "aspect_sentiment_analysis":
               teacher_output = item["target_text"]
               model_output = generate_response(model, self.tokenizer,item["prompt_text"], max_new_tokens=self.max_new_tokens)
               teacher_sentiment = extract_sentiment(teacher_output)
               student_sentiment = extract_sentiment(model_output)
               teacher_sentiments.append(teacher_sentiment)
               student_sentiments.append(student_sentiment)
            elif item["task"] == "aspect_emotion_detection":
               teacher_output = item["target_text"]
               model_output = generate_response(model, self.tokenizer,item["prompt_text"], max_new_tokens=self.max_new_tokens)
               teacher_emotion = extract_emotion(teacher_output)
               student_emotion = extract_emotion(model_output)
               teacher_emotions.append(teacher_emotion)
               student_emotions.append(student_emotion)

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
      
      print(f"Results")
      for k, v in metrics.items():
         print(f"{k}: {v:.4f}")
      
      trainer = kwargs.get("trainer", None)
      if trainer is not None:
         trainer.log(metrics)

      model.train()
      return control