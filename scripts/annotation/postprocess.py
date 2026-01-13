from parsing import extract_sentiment, extract_emotion
import json

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


for file in files_to_check:
   with open(file, "r") as f:
      data = json.load(f)

   for idx, tweet in enumerate(data.keys()):
      tweet_data = data[tweet]
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
      json.dump(data, f)