SYSTEM_ASPECT_EXTRACTION = (
    "You are an NLP expert specializing in aspect-based sentiment analysis. "
    "Your task is to identify aspect terms (entities, topics, or features) "
    "that people express opinions about. "
    "The reply length is limited to 150 words or less."
)

SYSTEM_SYNTACTIC = (
    "You are an AI assistant that helps people find information. "
    "Please refine your reply and ensure its accuracy. "
    "The reply length is limited to 200 words or less."
)


SYSTEM_OPINION = (
    "You are an AI assistant that helps people find information. "
    "Please refine your reply and ensure its accuracy. "
    "The reply length is limited to 120 words or less."
)

SYSTEM_SENTIMENT = (
    "You are a sentiment analysis expert. "
    "Given a sentence and an aspect, determine the sentiment polarity "
    "of that aspect as one of: positive, negative, or neutral. "
    "Always answer in the format:\n"
    "The sentiment towards <aspect> in the given sentence is <positive|negative|neutral>. Because <reason>."
)


SYSTEM_EMOTION = (
    "You are an emotion analysis expert. "
    "Your task is to identify the emotions expressed towards specific aspects in text. "
    "The reply length is limited to 120 words or less."
)


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

EMOTION_LABELS = [
    "optimistic",
    "thankful",
    "empathetic",
    "pessimistic",
    "anxious",
    "sad",
    "annoyed",
    "hopeful",
    "proud",
    "trustful",
    "satisfied",
    "scared",
    "angry",
    "no_emotion",
    "neutral"
]


def get_emotion_labels_str():
    return ", ".join(EMOTION_LABELS[:-1]) + f", or {EMOTION_LABELS[-1]}"


def prompt_aspect_extraction(sentences: list, structures: list) -> list:
    prompts = []
    
    for sentence, structure in zip(sentences, structures):
        prompt = (
            f"Given the sentence '{sentence}', "
            f'and its syntactic structure: "{structure}" '
            f"Identify all aspect terms in this sentence. Aspect terms are entities, "
            f"topics, or features that the speaker expresses an opinion about. Ensure that aspect is the very name of entity, topic of feature present in the text."
            f"For COVID-19 related tweets, aspects might include: vaccines, lockdowns, "
            f"masks, government response, healthcare workers, symptoms, etc. "
            f"\n\nFormat your response as follows:\n"
            f"ASPECT: <aspect_term>\n"
            f"REASON: <explanation>\n\n"
            f"Repeat this format for each aspect found. Each aspect must be word person expresses opinion towards in COVID times."
        )

        message =  [
            {"role": "system", "content": SYSTEM_ASPECT_EXTRACTION},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts

def prompt_aspect_extraction_no_res(sentences: list, structures: list) -> list:
    prompts = []
    
    for sentence, structure in zip(sentences, structures):
        prompt = (
            f"Given the sentence '{sentence}', "
            f'and its syntactic structure: "{structure}" '
            f"Identify all aspect terms in this sentence. Aspect terms are entities, "
            f"topics, or features that the speaker expresses an opinion about. Ensure that aspect is the very name of entity, topic of feature present in the text."
            f"For COVID-19 related tweets, aspects might include: vaccines, lockdowns, "
            f"masks, government response, healthcare workers, symptoms, etc. "
            f"\n\nFormat your response as follows:\n"
            f"ASPECT: <aspect_term>\n"
            f"Repeat this format for each aspect found. Each aspect must be word person expresses opinion towards in COVID times."
        )

        message =  [
            {"role": "system", "content": SYSTEM_ASPECT_EXTRACTION},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts

def prompt_aspect_extraction_labeled(sentences: list, structures: list, aspects_list: list ) -> list:
    prompts = []
    
    for sentence, structure, aspects in zip(sentences, structures, aspects_list):
        aspects_str = ", ".join([f"'{a}'" for a in aspects])

        prompt = (
            f"Given the sentence '{sentence}', "
            f'and its syntactic structure: "{structure}" '
            f"Identify all aspect terms in this sentence. Aspect terms are entities, "
            f"topics, or features that the speaker expresses an opinion about. "
            f"The aspect terms in this sentence are: {aspects_str}. "
            f"Explain why each of these is an aspect term.  Each aspect must be word person expresses opinion towards in COVID times."
        )

        message = [
            {"role": "system", "content": SYSTEM_ASPECT_EXTRACTION},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_syntactic_parsing(sentences: list, aspects: list, structures: list) -> list:
    prompts = []
    
    for sentence, aspect, structure in zip(sentences, aspects, structures):
        prompt = (
            f"Given the sentence '{sentence}', "
            f'"{structure}" is the CoNLL-U format for the syntactic dependency '
            f"relationship of this sentence. {CONLL_EXPLANATION} "
            f"Based on the syntactic dependency information of the sentence, "
            f"analyze information related to '{aspect}' in the sentence."
        )

        message = [
            {"role": "system", "content": SYSTEM_SYNTACTIC},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_syntactic_parsing_labeled(sentences: list, aspects: list, structures: list, polarities: list
) -> list:
    prompts = []
    
    for sentence, aspect, structure, polarity in zip(sentences, aspects, structures, polarities):
        prompt = (
            f"Given the sentence '{sentence}', "
            f'"{structure}" is the CoNLL-U format for the syntactic dependency '
            f"relationship of this sentence. {CONLL_EXPLANATION} "
            f"Based on the syntactic dependency information of the sentence, "
            f"analyze information related to '{aspect}' in the sentence. "
            f"The sentiment polarity towards '{aspect}' is {polarity}."
        )

        message = [
            {"role": "system", "content": SYSTEM_SYNTACTIC},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_opinion_extraction(sentences: list, aspects: list, syntactic_infos: list) -> list:
    prompts = []
    
    for sentence, aspect, syntactic_info in zip(sentences, aspects, syntactic_infos):
        prompt = (
            f"Given the sentence '{sentence}', "
            f"{syntactic_info} "
            f"Considering the context and information related to '{aspect}', "
            f"what is the speaker's opinion towards '{aspect}'?"
        )

        message = [
            {"role": "system", "content": SYSTEM_OPINION},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_opinion_extraction_labeled(sentences: list, aspects: list, syntactic_infos: list, polarities: list
) -> list:
    prompts = []
    
    for sentence, aspect, syntactic_info, polarity in zip(sentences, aspects, syntactic_infos, polarities):
        prompt = (
            f"Given the sentence '{sentence}', "
            f"{syntactic_info} "
            f"Considering the context and information related to '{aspect}', "
            f"what is the speaker's opinion towards '{aspect}'? "
            f"The sentiment polarity towards '{aspect}' is {polarity}."
        )

        message = [
            {"role": "system", "content": SYSTEM_OPINION},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_emotion_classification(sentences: list, aspects: list, opinion_infos: list, sentiments: list
) -> list:
    prompts = []
    emotion_options = get_emotion_labels_str()
    
    for sentence, aspect, opinion_info, sentiment in zip(sentences, aspects, opinion_infos, sentiments):
        prompt = (
            f"Given the sentence '{sentence}', "
            f"and the opinion analysis: {opinion_info} "
            f"and this sentiment label: {sentiment} "
            f"What emotion is the speaker expressing towards '{aspect}'? "
            f"Choose exactly one label from: {emotion_options}. "
            f"First output: Emotion: <label>\n. Then explain your reasoning. You must follow this format."
        )

        message = [
            {"role": "system", "content": SYSTEM_EMOTION},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_emotion_classification_labeled(sentences: list, aspects: list, opinion_infos: list, emotions: list
) -> list:
    prompts = []
    emotion_options = get_emotion_labels_str()
    
    for sentence, aspect, opinion_info, emotion in zip(sentences, aspects, opinion_infos, emotions):
        prompt = (
            f"Given the sentence '{sentence}', "
            f"and the opinion analysis: {opinion_info} "
            f"What emotion is the speaker expressing towards '{aspect}'? "
            f"Choose exactly one emotion from: {emotion_options}. "
            f"The emotion expressed towards '{aspect}' is {emotion}. "
            f"Explain why this emotion is appropriate based on the language used."
            f"First output: Emotion: <label>\n. Then explain your reasoning. You must follow this format."
        )

        message = [
            {"role": "system", "content": SYSTEM_EMOTION},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_sentiment_classification(sentences: list, aspects: list, opinion_infos: list
) -> list:
    prompts = []
    
    for sentence, aspect, opinion_info in zip(sentences, aspects, opinion_infos):
        prompt = (
            f"Given the sentence '{sentence}', "
            f"{opinion_info} "
            f"Based on common sense and the speaker's opinion, "
            f"what is the sentiment polarity towards '{aspect}'?"
            f"First output: Sentiment: <label>\n. Then explain your reasoning. You must follow this format."
        )

        message = [
            {"role": "system", "content": SYSTEM_SENTIMENT},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts


def prompt_sentiment_classification_labeled(sentences: list, aspects: list, opinion_infos: list, polarities: list
) -> list:
    prompts = []
    
    for sentence, aspect, opinion_info, polarity in zip(sentences, aspects, opinion_infos, polarities):
        prompt = (
            f"Given the sentence '{sentence}', "
            f"{opinion_info} "
            f"Based on common sense and the speaker's opinion, "
            f"what is the sentiment polarity towards '{aspect}'? "
            f"The sentiment polarity towards '{aspect}' is {polarity}."
            f"First output: Sentiment: <label>\n. Then explain your reasoning. You must follow this format."
        )

        message = [
            {"role": "system", "content": SYSTEM_SENTIMENT},
            {"role": "user", "content": prompt},
        ]
        prompts.append(message)
    return prompts
