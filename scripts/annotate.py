#!/usr/bin/env python3
"""
Generate reasoning traces for CONVERSATIONAL tweets only (filters out news headlines).
Converted from 02_Teacher_Annotation.ipynb for batch execution.
"""

import pandas as pd
import torch
import re
from tqdm import tqdm
from annotation.qwen_model import QwenTeacher
from annotation.prompts import (
    prompt_syntactic_parsing,
    prompt_aspect_extraction,
    prompt_opinion_extraction,
    prompt_emotion_classification,
    prompt_sentiment_classification,
)

# Configuration
N_SAMPLES = 50  # Target number of CONVERSATIONAL tweets (TEST RUN)
INPUT_FILE = "data/COVIDSenti/COVIDSenti_full_parsed.csv"
OUTPUT_FILE = f"data/COVIDSenti/COVIDSenti_conversational_annotated_{N_SAMPLES}.csv"

print("=" * 60)
print("QWEN TEACHER ANNOTATION - CONVERSATIONAL TWEETS ONLY")
print("=" * 60)

# Check GPU
print(f"\nNode: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected! This script requires a GPU node.")


# News filter
def is_news_like(tweet):
    """
    Detect if a tweet looks like a news headline vs conversational.

    Returns True if news-like (filter out), False if conversational (keep).
    """
    tweet_lower = tweet.lower()

    # Strong news indicators (automatic disqualification)
    strong_news_patterns = [
        r'^[A-Z][a-z]+ \| ',  # Title | Format (e.g., "Coronavirus | CDC")
        r'\bvia @',  # "via @username" (shared news)
        r'^RT @',  # Retweets
        r'^[A-Z][a-z\s]+ on [A-Z][a-z]+:',  # "Organization on Topic:" (e.g., "Wyoming Public Health on Coronavirus:")
        r':\s*["\u201c\u2018]',  # Colon followed by quote mark (quoted statements)
    ]

    # News hashtags (news aggregators and breaking news)
    news_hashtags = [
        r'#smartnews', r'#breakingnews', r'#breaking', r'#news',
        r'#topstory', r'#headline', r'#update', r'#alert',
        r'#cnn', r'#fox', r'#bbc', r'#msnbc', r'#reuters'
    ]

    for pattern in strong_news_patterns:
        if re.search(pattern, tweet):
            return True

    # Check for news hashtags
    for hashtag in news_hashtags:
        if hashtag in tweet_lower:
            return True

    # Check for first-person pronouns FIRST (strong conversational signals)
    # Do this before length check to catch short conversational tweets like "I trust vaccines"
    first_person = ['i ', 'my ', 'me ', "i'm", "i've", "i'd", "i'll",
                    'we ', 'our ', "we're", "we've", "we'll"]
    has_first_person = any(word in tweet_lower for word in first_person)

    # Check for questions FIRST (strong conversational signals)
    has_question = '?' in tweet

    # If has strong conversational signal, keep it (even if short)
    if has_first_person or has_question:
        return False

    # Very short tweets (< 5 words) WITHOUT conversational signals are likely headlines
    if len(tweet.split()) < 5:
        return True

    # Headline patterns (third-person narrative, passive voice)
    # e.g., "Washington state man who traveled..." or "X reports that..."
    headline_patterns = [
        r'^[A-Z][a-z\s]+ (man|woman|person|official|doctor|patient|resident)',  # "State/City man/woman..."
        r'\b(reports?|says?|confirms?|announces?|warns?|urges?)\s+(that|about)',  # "X reports that..."
        r'^\w+\s+(is|was|has been|have been)\s+the\s+(first|second|latest)',  # "X is the first..."
    ]

    for pattern in headline_patterns:
        if re.search(pattern, tweet):
            return True

    # Check for second-person (weaker signal - could be in quotes/directives)
    second_person = ['you ', 'your ', "you're", "you've", "you'll"]
    has_second_person = any(word in tweet_lower for word in second_person)

    # Emotional/conversational punctuation
    has_exclamation = '!' in tweet

    # Opinion/emotion words
    opinion_words = ['think', 'feel', 'believe', 'hope', 'wish', 'hate', 'love',
                     'like', 'dislike', 'want', 'need', 'afraid', 'worried',
                     'glad', 'happy', 'sad', 'angry', 'confused', 'admit',
                     'crap', 'damn', 'wow', 'omg', 'wtf', 'lol', 'lmao']
    has_opinion = any(word in tweet_lower for word in opinion_words)

    # Check for URLs
    has_url = bool(re.search(r'https?://', tweet))

    # Decision logic (continuing from strong signals checked above):
    # 1. Opinion words + exclamation = conversational
    if has_opinion and has_exclamation:
        return False

    # 2. Second-person + opinion (not just directive) = conversational
    if has_second_person and has_opinion:
        return False

    # 6. If has URL but no strong conversational signals = likely news
    if has_url:
        return True

    # 7. Institutional mentions without personal context = news
    institutional = bool(re.search(r'\b(CDC|WHO|NIH|FDA|Health Department|Public Health)\b', tweet, re.IGNORECASE))
    if institutional:
        return True

    # 8. Default: if no conversational signals = news-like
    return True


# Load and filter data
print(f"\nLoading data from {INPUT_FILE}...")
df_all = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df_all):,} tweets")

# Filter to conversational tweets
print("\nFiltering for conversational tweets (excluding news)...")
df_all['is_news'] = df_all['tweet'].apply(is_news_like)
df_conversational = df_all[~df_all['is_news']].copy()

print(f"  News-like tweets: {df_all['is_news'].sum():,}")
print(f"  Conversational tweets: {len(df_conversational):,}")

# Take subset of conversational tweets
df_subset = df_conversational.head(N_SAMPLES).copy()
print(f"\nProcessing {len(df_subset)} conversational tweets\n")

# Load teacher model
print("Loading Qwen2.5-72B-Instruct teacher model...")
teacher = QwenTeacher()
print()


# Annotation function
def annotate_tweet(teacher, sentence, parse, dummy_aspect="coronavirus"):
    """Generate all 5 reasoning traces for a single tweet."""
    r1 = teacher.generate_reasoning_trace(
        prompt_syntactic_parsing(sentence, dummy_aspect, parse), max_new_tokens=200
    )
    r2 = teacher.generate_reasoning_trace(
        prompt_aspect_extraction(sentence, parse), max_new_tokens=150
    )
    r3 = teacher.generate_reasoning_trace(
        prompt_opinion_extraction(sentence, dummy_aspect, r1), max_new_tokens=120
    )
    r4 = teacher.generate_reasoning_trace(
        prompt_sentiment_classification(sentence, dummy_aspect, r3), max_new_tokens=120
    )
    r5 = teacher.generate_reasoning_trace(
        prompt_emotion_classification(sentence, dummy_aspect, r3), max_new_tokens=120
    )

    return {
        'r1_syntactic': r1,
        'r2_aspects': r2,
        'r3_opinion': r3,
        'r4_sentiment': r4,
        'r5_emotion': r5,
    }


# Annotate tweets
print("Generating reasoning traces for conversational tweets...")
results = []
errors = 0

for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    try:
        annotations = annotate_tweet(teacher, row['tweet'], row['conllu_parse'])
        results.append(annotations)
    except Exception as e:
        print(f"\nError at index {idx}: {e}")
        errors += 1
        results.append({
            'r1_syntactic': '',
            'r2_aspects': '',
            'r3_opinion': '',
            'r4_sentiment': '',
            'r5_emotion': '',
        })

# Save results
df_annotated = pd.concat([df_subset.reset_index(drop=True), pd.DataFrame(results)], axis=1)
df_annotated.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'=' * 60}")
print("COMPLETE")
print(f"{'=' * 60}")
print(f"Conversational tweets annotated: {len(df_annotated)}")
print(f"Errors: {errors}")
print(f"Saved to: {OUTPUT_FILE}")

# Show sample
print(f"\nSample conversational annotation:")
print(f"Tweet: {df_annotated.iloc[0]['tweet']}")
print(f"Label: {df_annotated.iloc[0]['label']}")
print(f"\nR2 (Aspects): {df_annotated.iloc[0]['r2_aspects'][:150]}...")
