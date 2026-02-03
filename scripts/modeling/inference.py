import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

from scripts.qwen_model.qwen_model import generate_batch
from scripts.qwen_model.prompts import (
    prompt_aspect_extraction_no_res,
    prompt_syntactic_parsing,
    prompt_opinion_extraction,
    prompt_sentiment_classification,
    prompt_emotion_classification,
)
from scripts.annotation.parsing import extract_aspects, extract_sentiment, extract_emotion

def load_trained_model(
   base_model_path: str,
   checkpoint_path: str,
   use_8bit: bool = True,
   device_map: str = "auto",
):
   if use_8bit:
      bnb_config = BitsAndBytesConfig(
         load_in_8bit=True,
         load_in_4bit=False,
         llm_int8_enable_fp32_cpu_offload=True, 
      )
   else:
      bnb_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_quant_type="nf4",
         bnb_4bit_compute_dtype=torch.bfloat16,
         bnb_4bit_use_double_quant=True,
      )
   
   tokenizer = AutoTokenizer.from_pretrained(
      base_model_path,
      local_files_only=True,
      padding_side='left'
   )
   if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
   
   base_model = AutoModelForCausalLM.from_pretrained(
      base_model_path,
      local_files_only=True,
      dtype=torch.bfloat16,
      quantization_config=bnb_config,
      device_map=device_map,
   )
   
   model = PeftModel.from_pretrained(
      base_model,
      checkpoint_path,
      is_trainable=False,  
   )
   
   model.eval()
   return model, tokenizer

def run_aspect_extraction(model, tokenizer, text: str, conllu_parse: str, max_new_tokens: int = 256):
   prompts = prompt_aspect_extraction_no_res([text], [conllu_parse])
   outputs = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size=1)
   aspects = extract_aspects(outputs[0])
   return aspects, outputs[0]


def run_syntactic_parsing(model, tokenizer, text: str, aspect: str, conllu_parse: str, max_new_tokens: int = 256):
   prompts = prompt_syntactic_parsing([text], [aspect], [conllu_parse])
   outputs = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size=1)
   return outputs[0]


def run_opinion_extraction(model, tokenizer, text: str, aspect: str, syntactic_info: str, max_new_tokens: int = 256):
   prompts = prompt_opinion_extraction([text], [aspect], [syntactic_info])
   outputs = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size=1)
   return outputs[0]


def run_sentiment_analysis(model, tokenizer, text: str, aspect: str, opinion_info: str, max_new_tokens: int = 256):
   prompts = prompt_sentiment_classification([text], [aspect], [opinion_info])
   outputs = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size=1)
   sentiment_label = extract_sentiment(outputs[0])
   return sentiment_label, outputs[0]


def run_emotion_detection(model, tokenizer, text: str, aspect: str, opinion_info: str, sentiment_label: str, max_new_tokens: int = 256):
   prompts = prompt_emotion_classification([text], [aspect], [opinion_info], [sentiment_label])
   outputs = generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size=1)
   emotion_label = extract_emotion(outputs[0])
   return emotion_label, outputs[0]


def run_full_pipeline(model, tokenizer, text: str, conllu_parse: str, max_new_tokens: int = 256):
   result = {
      "text": text,
      "aspects": {},
      "aspect_syntactic": {},
      "aspect_opinions": {},
      "aspect_sentiments_raw": {},
      "aspect_sentiments_label": {},
      "aspect_emotions_raw": {},
      "aspect_emotions_label": {},
   }
   
   aspects, raw_aspect_output = run_aspect_extraction(model, tokenizer, text, conllu_parse, max_new_tokens)
   result["raw_aspect_extraction_output"] = raw_aspect_output
   
   for idx, aspect in enumerate(aspects):
      result["aspects"][aspect] = idx
      k = str(idx)
      
      syntactic_info = run_syntactic_parsing(model, tokenizer, text, aspect, conllu_parse, max_new_tokens)
      result["aspect_syntactic"][k] = syntactic_info
      
      opinion_info = run_opinion_extraction(model, tokenizer, text, aspect, syntactic_info, max_new_tokens)
      result["aspect_opinions"][k] = opinion_info
      
      sentiment_label, sentiment_raw = run_sentiment_analysis(model, tokenizer, text, aspect, opinion_info, max_new_tokens)
      result["aspect_sentiments_label"][k] = sentiment_label
      result["aspect_sentiments_raw"][k] = sentiment_raw
      
      emotion_label, emotion_raw = run_emotion_detection(model, tokenizer, text, aspect, opinion_info, sentiment_label, max_new_tokens)
      result["aspect_emotions_label"][k] = emotion_label
      result["aspect_emotions_raw"][k] = emotion_raw
   
   return result

BASE_MODEL_PATH = "/home/jovyan/models/Meta-Llama-3-8B-Instruct"
CHECKPOINT_PATH = "/home/jovyan/synchain-absa-emotion/trained_models/checkpoint-570"
MAX_NEW_TOKENS = 256

llama_model, llama_tokenizer = load_trained_model(BASE_MODEL_PATH, CHECKPOINT_PATH)

conllu_parse = "# text = China has more cases of coronavirus than it had of SARS, but a vaccine could be on the horizon #Topbuzz\n1\tChina\tChina\tPROPN\tNNP\t_\t2\tnsubj\t_\t_\n2\thas\thave\tVERB\tVBZ\t_\t0\tROOT\t_\t_\n3\tmore\tmore\tADJ\tJJR\t_\t4\tamod\t_\t_\n4\tcases\tcase\tNOUN\tNNS\t_\t2\tdobj\t_\t_\n5\tof\tof\tADP\tIN\t_\t4\tprep\t_\t_\n6\tcoronavirus\tcoronavirus\tNOUN\tNN\t_\t5\tpobj\t_\t_\n7\tthan\tthan\tSCONJ\tIN\t_\t9\tmark\t_\t_\n8\tit\tit\tPRON\tPRP\t_\t9\tnsubj\t_\t_\n9\thad\thave\tVERB\tVBD\t_\t2\tadvcl\t_\t_\n10\tof\tof\tADP\tIN\t_\t9\tprep\t_\t_\n11\tSARS\tSARS\tPROPN\tNNP\t_\t10\tpobj\t_\t_\n12\t,\t,\tPUNCT\t,\t_\t2\tpunct\t_\t_\n13\tbut\tbut\tCCONJ\tCC\t_\t2\tcc\t_\t_\n14\ta\ta\tDET\tDT\t_\t15\tdet\t_\t_\n15\tvaccine\tvaccine\tNOUN\tNN\t_\t17\tnsubj\t_\t_\n16\tcould\tcould\tAUX\tMD\t_\t17\taux\t_\t_\n17\tbe\tbe\tAUX\tVB\t_\t2\tconj\t_\t_\n18\ton\ton\tADP\tIN\t_\t17\tprep\t_\t_\n19\tthe\tthe\tDET\tDT\t_\t20\tdet\t_\t_\n20\thorizon\thorizon\tNOUN\tNN\t_\t18\tpobj\t_\t_\n21\t#\t#\tSYM\t$\t_\t17\tpunct\t_\t_\n22\tTopbuzz\tTopbuzz\tPROPN\tNNP\t_\t17\tattr\t_\t_"
tweet = "China has more cases of coronavirus than it had of SARS, but a vaccine could be on the horizon #Topbuzz"
result = run_full_pipeline(llama_model, llama_tokenizer, tweet, conllu_parse, MAX_NEW_TOKENS)

print(result)