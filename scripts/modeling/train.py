import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..annotation.prompts import (
   prompt_aspect_extraction,
   prompt_syntactic_parsing,
   prompt_opinion_extraction,
   prompt_sentiment_classification,
   prompt_emotion_classification,
   EMOTION_LABELS
)


def parse_aspect_extraction(text):
   aspects = []
   reasoning = text.strip()
   
   lines = text.strip().split('\n')
   for line in lines:
      if any(keyword in line.lower() for keyword in ['aspect:', 'aspects:', '-', '*']):
         potential_aspects = re.findall(r"['\"]([^'\"]+)['\"]", line)
         aspects.extend(potential_aspects)
   
   return aspects, reasoning

def parse_syntactic_info(text):
   return text.strip()

def parse_opinion(text):
   return text.strip()

def parse_sentiment(text):
   text_lower = text.lower()
   
   if 'positive' in text_lower:
      label = 'positive'
   elif 'negative' in text_lower:
      label = 'negative'
   elif 'neutral' in text_lower:
      label = 'neutral'
   else:
      label = 'neutral'
   
   reasoning = text.strip()
   return label, reasoning

def parse_emotion(text):
   text_lower = text.lower()
   
   label = None
   for emotion in EMOTION_LABELS:
      if emotion.lower() in text_lower:
         label = emotion
         break
   
   if label is None:
      label = EMOTION_LABELS[0]
   
   reasoning = text.strip()
   return label, reasoning

def knowledge_distillation(
   student_model, 
   student_tokenizer, 
   teacher_annotated_data, 
   original_sentences, 
   conllu_parses, 
   max_new_tokens=512
):
   print("Task 1: Aspect Extraction")
   student_aspects = []
   student_aspect_extraction_reasoning = []
   
   for i, sentence in enumerate(tqdm.tqdm(original_sentences)):
      conllu_parse = conllu_parses[i]
      
      messages = prompt_aspect_extraction(sentence=sentence, structure=conllu_parse)
      response = generate_response(student_model, student_tokenizer, messages, max_new_tokens=max_new_tokens)
      
      aspects, reasoning = parse_aspect_extraction(response)
      
      student_aspects.append(aspects)
      student_aspect_extraction_reasoning.append(reasoning)
   
   print("Task 2: Syntactic Parsing")
   teacher_aspects = teacher_annotated_data["aspects"]
   student_syntactic_aspect_info = []
   
   for i, sentence in enumerate(tqdm.tqdm(original_sentences)):
      conllu_parse = conllu_parses[i]
      teacher_aspects_sentence = teacher_aspects[i]
      
      student_syntactic_aspect_info_sentence = []
      
      for aspect in teacher_aspects_sentence:
         messages = prompt_syntactic_parsing(sentence=sentence, aspect=aspect, structure=conllu_parse)
         response = generate_response(student_model, student_tokenizer, messages, max_new_tokens=max_new_tokens)
         
         syntactic_reasoning = parse_syntactic_info(response)
         student_syntactic_aspect_info_sentence.append(syntactic_reasoning)
      
      student_syntactic_aspect_info.append(student_syntactic_aspect_info_sentence)
   
   print("Task 3: Opinion Extraction")
   teacher_syntactic_aspect_info = teacher_annotated_data["aspect_syntactic_info"]
   student_opinions = []
   
   for i, sentence in enumerate(tqdm.tqdm(original_sentences)):
      teacher_aspects_sentence = teacher_aspects[i]
      teacher_syntactic_aspect_info_sentence = teacher_syntactic_aspect_info[i]
      
      student_opinions_sentence = []
      
      for j, aspect in enumerate(teacher_aspects_sentence):
         syntactic_context = teacher_syntactic_aspect_info_sentence[j]
         
         messages = prompt_opinion_extraction(sentence=sentence, aspect=aspect, syntactic_info=syntactic_context)
         response = generate_response(student_model, student_tokenizer, messages, max_new_tokens=max_new_tokens)
         
         opinion_reasoning = parse_opinion(response)
         student_opinions_sentence.append(opinion_reasoning)
      
      student_opinions.append(student_opinions_sentence)
   
   print("Task 4: Sentiment Classification")
   teacher_opinions = teacher_annotated_data["opinions"]
   student_sentiments_labels = []
   student_sentiments_reasoning = []
   
   for i, sentence in enumerate(tqdm.tqdm(original_sentences)):
      teacher_aspects_sentence = teacher_aspects[i]
      teacher_opinions_sentence = teacher_opinions[i]
      
      student_sentiments_labels_sentence = []
      student_sentiments_reasoning_sentence = []
      
      for j, aspect in enumerate(teacher_aspects_sentence):
         opinion_context = teacher_opinions_sentence[j]
         
         messages = prompt_sentiment_classification(sentence, aspect, opinion_context)
         response = generate_response(student_model, student_tokenizer, messages, max_new_tokens=max_new_tokens)
         
         sentiment_label, sentiment_reasoning = parse_sentiment(response)
         
         student_sentiments_labels_sentence.append(sentiment_label)
         student_sentiments_reasoning_sentence.append(sentiment_reasoning)
      
      student_sentiments_labels.append(student_sentiments_labels_sentence)
      student_sentiments_reasoning.append(student_sentiments_reasoning_sentence)
   
   print("Task 5: Emotion Classification")
   student_emotions_labels = []
   student_emotions_reasoning = []
   
   for i, sentence in enumerate(tqdm.tqdm(original_sentences)):
      teacher_aspects_sentence = teacher_aspects[i]
      teacher_opinions_sentence = teacher_opinions[i]
      
      student_emotions_labels_sentence = []
      student_emotions_reasoning_sentence = []
      
      for j, aspect in enumerate(teacher_aspects_sentence):
         opinion_context = teacher_opinions_sentence[j]
         
         messages = prompt_emotion_classification(sentence, aspect, opinion_context)
         response = generate_response(student_model, student_tokenizer, messages, max_new_tokens=max_new_tokens)
         
         emotion_label, emotion_reasoning = parse_emotion(response)
         
         student_emotions_labels_sentence.append(emotion_label)
         student_emotions_reasoning_sentence.append(emotion_reasoning)
      
      student_emotions_labels.append(student_emotions_labels_sentence)
      student_emotions_reasoning.append(student_emotions_reasoning_sentence)
   
   return {
      "student_aspects": student_aspects,
      "student_aspect_extraction_reasoning": student_aspect_extraction_reasoning,
      "student_syntactic_aspect_info": student_syntactic_aspect_info,
      "student_opinions": student_opinions,
      "student_sentiments_labels": student_sentiments_labels,
      "student_sentiments_reasoning": student_sentiments_reasoning,
      "student_emotions_labels": student_emotions_labels,
      "student_emotions_reasoning": student_emotions_reasoning,
   }
   

MAX_NEW_TOKENS = 512
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

student_results = knowledge_distillation(
   student_model= model, 
   student_tokenizer= tokenizer, 
   teacher_annotated_data=..., 
   original_sentences=..., 
   conllu_parses=..., 
   max_new_tokens=MAX_NEW_TOKENS
)