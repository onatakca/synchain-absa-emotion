import re

def is_news_like(tweet):
   tweet_lower = tweet.lower()

   strong_news_patterns = [
      r'^[A-Z][a-z]+ \| ',
      r'\bvia @',
      r'^RT @',
      r'^[A-Z][a-z\s]+ on [A-Z][a-z]+:',
      r':\s*["\u201c\u2018]',
   ]

   news_hashtags = [
      r'#smartnews', r'#breakingnews', r'#breaking', r'#news',
      r'#topstory', r'#headline', r'#update', r'#alert',
      r'#cnn', r'#fox', r'#bbc', r'#msnbc', r'#reuters'
   ]

   for pattern in strong_news_patterns:
      if re.search(pattern, tweet):
         return True

   for hashtag in news_hashtags:
      if hashtag in tweet_lower:
         return True

   first_person = ['i ', 'my ', 'me ', "i'm", "i've", "i'd", "i'll",
                  'we ', 'our ', "we're", "we've", "we'll"]
   has_first_person = any(word in tweet_lower for word in first_person)

   has_question = '?' in tweet

   if has_first_person or has_question:
      return False

   if len(tweet.split()) < 5:
      return True

   headline_patterns = [
      r'^[A-Z][a-z\s]+ (man|woman|person|official|doctor|patient|resident)',
      r'\b(reports?|says?|confirms?|announces?|warns?|urges?)\s+(that|about)',
      r'^\w+\s+(is|was|has been|have been)\s+the\s+(first|second|latest)',
   ]

   for pattern in headline_patterns:
      if re.search(pattern, tweet):
         return True

   second_person = ['you ', 'your ', "you're", "you've", "you'll"]
   has_second_person = any(word in tweet_lower for word in second_person)

   has_exclamation = '!' in tweet

   opinion_words = ['think', 'feel', 'believe', 'hope', 'wish', 'hate', 'love',
                  'like', 'dislike', 'want', 'need', 'afraid', 'worried',
                  'glad', 'happy', 'sad', 'angry', 'confused', 'admit',
                  'crap', 'damn', 'wow', 'omg', 'wtf', 'lol', 'lmao']
   has_opinion = any(word in tweet_lower for word in opinion_words)

   has_url = bool(re.search(r'https?://', tweet))

   if has_opinion and has_exclamation:
      return False

   if has_second_person and has_opinion:
      return False

   if has_url:
      return True

   institutional = bool(re.search(r'\b(CDC|WHO|NIH|FDA|Health Department|Public Health)\b', tweet, re.IGNORECASE))
   if institutional:
      return True

   return True

def extract_aspects(text):
   """
   Extract aspects from     format:
   ASPECT: <aspect_term>
   REASON: <explanation>
   """
   text = text.strip()
   if not text:
      return []
   
   aspects = []
   matches = re.findall(r'ASPECT:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
   for match in matches:
      aspect = match.strip().lower()
      if aspect:
         aspects.append(aspect)
   
   if not aspects:
      print("Issue with aspect parse from the LLM output.")
      print(text)
      parts = re.split(r"[,;\n]", text)
      aspects = [p.strip().lower() for p in parts if len(p.strip()) > 0]
   
   return aspects
