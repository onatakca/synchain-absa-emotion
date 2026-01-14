from parsing import extract_sentiment, extract_emotion
import json
import pandas as pd
import os

files_to_check = [
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covid19nlp_chunk0_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk0_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk1_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk2_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk3_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk4_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk5_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk6_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk7_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk8_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk9_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk10_annotated.json",
   "/home/s3758869/synchain-absa-emotion/data/output_data/Qwen25-32b-instruct_annotation/covidsenti_chunk11_annotated.json",
]

conllu_parse_folder = "/home/s3758869/synchain-absa-emotion/data/input_data/chunks_for_teacher_model_ann"

for file in files_to_check:
   with open(file, "r") as f:
      data = json.load(f)

   file_name = file.split("/")[-1].replace("_annotated.json", ".csv")
   conllu_parse_file = pd.read_csv(os.path.join(conllu_parse_folder, file_name))
   
   for idx, tweet in enumerate(data.keys()):
      tweet_data = data[tweet]
      conllu_parse_data = conllu_parse_file.iloc[idx]
      data[tweet]["conllu_parse"] = conllu_parse_data["conllu_parse"]
      for aspect_num in tweet_data["aspect_sentiments_raw"]:
         aspect_num = str(aspect_num)
         sentiment = extract_sentiment(tweet_data["aspect_sentiments_raw"][aspect_num])
         if tweet_data["aspect_sentiments_label"][aspect_num] != sentiment:
            sent_changes+=1
            tweet_data["aspect_sentiments_label"][aspect_num] = sentiment
         emotion = extract_emotion(tweet_data["aspect_emotions_raw"][aspect_num])
         if emotion != tweet_data["aspect_emotions_label"][aspect_num]:
            tweet_data["aspect_emotions_label"][aspect_num] = emotion

   with open(file, "w") as f:
      json.dump(data, f, indent=3)
      


