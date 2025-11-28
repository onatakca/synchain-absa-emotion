"""
Prompt templates for COVID-19 ABSA with extended Syn-Chain.

Original Syn-Chain (3 steps):
    1. Syntactic Parsing
    2. Opinion Extraction  
    3. Sentiment Classification

Our Extension (5 tasks):
    1. Syntactic Parsing
    2. Aspect Extraction (NEW)
    3. Opinion Extraction
    4. Emotion Classification (NEW)
    5. Sentiment Classification

Each step has two versions:
    - Regular: For zero-shot inference
    - Label-conditioned: For generating training data (includes ground truth)
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_SYNTACTIC = (
    "You are an AI assistant that helps people find information. "
    "Please refine your reply and ensure its accuracy. "
    "The reply length is limited to 200 words or less."
)

SYSTEM_ASPECT_EXTRACTION = (
    "You are an NLP expert specializing in aspect-based sentiment analysis. "
    "Your task is to identify aspect terms (entities, topics, or features) "
    "that people express opinions about. "
    "The reply length is limited to 150 words or less."
)

SYSTEM_OPINION = (
    "You are an AI assistant that helps people find information. "
    "Please refine your reply and ensure its accuracy. "
    "The reply length is limited to 120 words or less."
)

SYSTEM_EMOTION = (
    "You are an emotion analysis expert. "
    "Your task is to identify the emotions expressed towards specific aspects in text. "
    "The reply length is limited to 120 words or less."
)

SYSTEM_SENTIMENT = (
    "You are a sentiment analysis expert. "
    "I will provide you with a sentence and a certain aspect mentioned in the sentence. "
    "Please analyze the sentiment polarity of that aspect in a given sentence. "
    "Output:\n'''\nThe sentiment towards {Aspect} in the given sentence is "
    "{positive, negative or neutral}. Because\n'''"
)

# =============================================================================
# CONLL-U EXPLANATION (used in syntactic parsing prompts)
# =============================================================================

CONLL_EXPLANATION = (
    "Each row in the table represents a word in the sentence, and each column "
    "represents some specific properties of the word, including: "
    "ID (word position in the sentence, starting from 1), "
    "TEXT (word itself), "
    "LEMMA (the base form of the word), "
    "POS (the simple UPOS part of speech tag), "
    "TAG (the detailed part of speech tag), "
    "FEATS (other grammatical features, blank here), "
    "HEAD (the header word of the dependency relationship of the current word, "
    "i.e. the ID of the word that the word depends on), "
    "DEPREL (dependency label, describing the relationship between the current "
    "word and the header word), "
    "DEPS (word dependency, empty here), "
    "MISC (other additional information, left blank here)."
)

# =============================================================================
# EMOTION LABELS (configurable - update based on your dataset)
# =============================================================================

# Default: NRC emotion categories commonly used for COVID-19 analysis
EMOTION_LABELS = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "trust",
    "anticipation",
    "surprise",
    "disgust",
]

def get_emotion_labels_str():
    """Returns formatted string of emotion labels for prompts."""
    return ", ".join(EMOTION_LABELS[:-1]) + f", or {EMOTION_LABELS[-1]}"


# =============================================================================
# STEP 1: SYNTACTIC PARSING
# =============================================================================

def prompt_syntactic_parsing(sentence: str, aspect: str, structure: str) -> list:
    """
    Step 1: Analyze syntactic dependencies related to the aspect.
    
    Args:
        sentence: The input sentence/tweet
        aspect: The aspect term to analyze
        structure: CoNLL-U formatted dependency parse
    
    Returns:
        List of message dicts for chat completion
    """
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'"{structure}" is the CoNLL-U format for the syntactic dependency '
        f'relationship of this sentence. {CONLL_EXPLANATION} '
        f'Based on the syntactic dependency information of the sentence, '
        f'analyze information related to \'{aspect}\' in the sentence.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_SYNTACTIC},
        {"role": "user", "content": prompt},
    ]


def prompt_syntactic_parsing_labeled(
    sentence: str, aspect: str, structure: str, polarity: str
) -> list:
    """
    Step 1 (labeled): For generating training data with ground truth.
    """
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'"{structure}" is the CoNLL-U format for the syntactic dependency '
        f'relationship of this sentence. {CONLL_EXPLANATION} '
        f'Based on the syntactic dependency information of the sentence, '
        f'analyze information related to \'{aspect}\' in the sentence. '
        f'The sentiment polarity towards \'{aspect}\' is {polarity}.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_SYNTACTIC},
        {"role": "user", "content": prompt},
    ]


# =============================================================================
# STEP 2: ASPECT EXTRACTION (NEW - your extension)
# =============================================================================

def prompt_aspect_extraction(sentence: str, structure: str) -> list:
    """
    Step 2: Extract aspect terms from the sentence.
    
    This is your NEW extension to Syn-Chain. It identifies what entities/topics
    people are expressing opinions about.
    
    Args:
        sentence: The input sentence/tweet
        structure: CoNLL-U formatted dependency parse
    
    Returns:
        List of message dicts for chat completion
    """
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'and its syntactic structure: "{structure}" '
        f'Identify all aspect terms in this sentence. Aspect terms are entities, '
        f'topics, or features that the speaker expresses an opinion about. '
        f'For COVID-19 related tweets, aspects might include: vaccines, lockdowns, '
        f'masks, government response, healthcare workers, symptoms, etc. '
        f'List each aspect term found and briefly explain why it is an aspect.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_ASPECT_EXTRACTION},
        {"role": "user", "content": prompt},
    ]


def prompt_aspect_extraction_labeled(
    sentence: str, structure: str, aspects: list
) -> list:
    """
    Step 2 (labeled): For generating training data with ground truth aspects.
    
    Args:
        aspects: List of ground truth aspect terms
    """
    aspects_str = ", ".join([f"'{a}'" for a in aspects])
    
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'and its syntactic structure: "{structure}" '
        f'Identify all aspect terms in this sentence. Aspect terms are entities, '
        f'topics, or features that the speaker expresses an opinion about. '
        f'The aspect terms in this sentence are: {aspects_str}. '
        f'Explain why each of these is an aspect term.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_ASPECT_EXTRACTION},
        {"role": "user", "content": prompt},
    ]


# =============================================================================
# STEP 3: OPINION EXTRACTION
# =============================================================================

def prompt_opinion_extraction(
    sentence: str, aspect: str, syntactic_info: str
) -> list:
    """
    Step 3: Extract the speaker's opinion towards the aspect.
    
    Args:
        sentence: The input sentence/tweet
        aspect: The aspect term to analyze
        syntactic_info: Output from Step 1 (syntactic parsing)
    
    Returns:
        List of message dicts for chat completion
    """
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'{syntactic_info} '
        f'Considering the context and information related to \'{aspect}\', '
        f'what is the speaker\'s opinion towards \'{aspect}\'?'
    )
    
    return [
        {"role": "system", "content": SYSTEM_OPINION},
        {"role": "user", "content": prompt},
    ]


def prompt_opinion_extraction_labeled(
    sentence: str, aspect: str, syntactic_info: str, polarity: str
) -> list:
    """
    Step 3 (labeled): For generating training data with ground truth.
    """
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'{syntactic_info} '
        f'Considering the context and information related to \'{aspect}\', '
        f'what is the speaker\'s opinion towards \'{aspect}\'? '
        f'The sentiment polarity towards \'{aspect}\' is {polarity}.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_OPINION},
        {"role": "user", "content": prompt},
    ]


# =============================================================================
# STEP 4: EMOTION CLASSIFICATION (NEW - your extension)
# =============================================================================

def prompt_emotion_classification(
    sentence: str, aspect: str, opinion_info: str
) -> list:
    """
    Step 4: Classify the emotion expressed towards the aspect.
    
    This is your NEW extension to Syn-Chain. Beyond positive/negative sentiment,
    it identifies specific emotions like fear, anger, trust, etc.
    
    Args:
        sentence: The input sentence/tweet
        aspect: The aspect term to analyze
        opinion_info: Output from Step 3 (opinion extraction)
    
    Returns:
        List of message dicts for chat completion
    """
    emotion_options = get_emotion_labels_str()
    
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'and the opinion analysis: {opinion_info} '
        f'What emotion is the speaker expressing towards \'{aspect}\'? '
        f'Choose from: {emotion_options}. '
        f'Explain your reasoning based on the language used and context.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_EMOTION},
        {"role": "user", "content": prompt},
    ]


def prompt_emotion_classification_labeled(
    sentence: str, aspect: str, opinion_info: str, emotion: str
) -> list:
    """
    Step 4 (labeled): For generating training data with ground truth emotion.
    """
    emotion_options = get_emotion_labels_str()
    
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'and the opinion analysis: {opinion_info} '
        f'What emotion is the speaker expressing towards \'{aspect}\'? '
        f'Choose from: {emotion_options}. '
        f'The emotion expressed towards \'{aspect}\' is {emotion}. '
        f'Explain why this emotion is appropriate based on the language used.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_EMOTION},
        {"role": "user", "content": prompt},
    ]


# =============================================================================
# STEP 5: SENTIMENT CLASSIFICATION
# =============================================================================

def prompt_sentiment_classification(
    sentence: str, aspect: str, opinion_info: str
) -> list:
    """
    Step 5: Classify the sentiment polarity towards the aspect.
    
    Args:
        sentence: The input sentence/tweet
        aspect: The aspect term to analyze
        opinion_info: Output from Step 3 (or Step 4 if using emotion)
    
    Returns:
        List of message dicts for chat completion
    """
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'{opinion_info} '
        f'Based on common sense and the speaker\'s opinion, '
        f'what is the sentiment polarity towards \'{aspect}\'?'
    )
    
    return [
        {"role": "system", "content": SYSTEM_SENTIMENT},
        {"role": "user", "content": prompt},
    ]


def prompt_sentiment_classification_labeled(
    sentence: str, aspect: str, opinion_info: str, polarity: str
) -> list:
    """
    Step 5 (labeled): For generating training data with ground truth.
    """
    prompt = (
        f'Given the sentence \'{sentence}\', '
        f'{opinion_info} '
        f'Based on common sense and the speaker\'s opinion, '
        f'what is the sentiment polarity towards \'{aspect}\'? '
        f'The sentiment polarity towards \'{aspect}\' is {polarity}.'
    )
    
    return [
        {"role": "system", "content": SYSTEM_SENTIMENT},
        {"role": "user", "content": prompt},
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_str_concat(*args) -> str:
    """Safely concatenate strings, handling None values."""
    return "".join([str(arg) if arg is not None else "" for arg in args])


def format_chain_output(
    syntactic_info: str,
    aspect_info: str,
    opinion_info: str,
    emotion_info: str,
    sentiment_info: str
) -> str:
    """
    Format the complete chain output for storage/debugging.
    
    Returns a formatted string with all chain steps labeled.
    """
    return safe_str_concat(
        "1-SYNTACTIC---", syntactic_info,
        "\n2-ASPECTS---", aspect_info,
        "\n3-OPINION---", opinion_info,
        "\n4-EMOTION---", emotion_info,
        "\n5-SENTIMENT---", sentiment_info
    )


# =============================================================================
# CONFIGURATION FOR DIFFERENT CHAIN STRATEGIES
# =============================================================================

# Based on the paper, different chain configurations can be used:
# - Full chain: 1⊕2⊕3⊕4⊕5 (all steps linked)
# - Break before sentiment: 1⊕2⊕3⊕4⊸5 (recommended by paper)
# - Break before emotion+sentiment: 1⊕2⊕3⊸4⊸5

CHAIN_CONFIGS = {
    "full": {
        "description": "All steps linked (1⊕2⊕3⊕4⊕5)",
        "use_syntactic_in_opinion": True,
        "use_opinion_in_emotion": True,
        "use_emotion_in_sentiment": True,
    },
    "break_sentiment": {
        "description": "Break before sentiment (1⊕2⊕3⊕4⊸5) - RECOMMENDED",
        "use_syntactic_in_opinion": True,
        "use_opinion_in_emotion": True,
        "use_emotion_in_sentiment": False,  # Break here
    },
    "break_emotion_sentiment": {
        "description": "Break before emotion and sentiment (1⊕2⊕3⊸4⊸5)",
        "use_syntactic_in_opinion": True,
        "use_opinion_in_emotion": False,  # Break here
        "use_emotion_in_sentiment": False,  # And here
    },
    "minimal": {
        "description": "Only syntactic linked to opinion (1⊕3⊸4⊸5)",
        "use_syntactic_in_opinion": True,
        "use_opinion_in_emotion": False,
        "use_emotion_in_sentiment": False,
    },
}


if __name__ == "__main__":
    # Example usage / testing
    test_sentence = "The vaccine rollout was slow but healthcare workers were amazing"
    test_aspect = "vaccine rollout"
    test_structure = "1\tThe\tthe\tDET\tDT\t_\t3\tdet\t_\t_\n..."  # truncated
    
    print("=== Testing Prompt Generation ===\n")
    
    # Test Step 1
    msgs = prompt_syntactic_parsing(test_sentence, test_aspect, test_structure)
    print("Step 1 (Syntactic Parsing):")
    print(f"  System: {msgs[0]['content'][:50]}...")
    print(f"  User: {msgs[1]['content'][:100]}...")
    print()
    
    # Test Step 2
    msgs = prompt_aspect_extraction(test_sentence, test_structure)
    print("Step 2 (Aspect Extraction):")
    print(f"  System: {msgs[0]['content'][:50]}...")
    print(f"  User: {msgs[1]['content'][:100]}...")
    print()
    
    # Test emotion labels
    print(f"Emotion labels: {get_emotion_labels_str()}")
    print()
    
    # Show chain configs
    print("Available chain configurations:")
    for name, config in CHAIN_CONFIGS.items():
        print(f"  {name}: {config['description']}")
