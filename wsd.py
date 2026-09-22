"""
wsd.py

Word Sense Disambiguation using the Lesk algorithm (NLTK's WordNet-based
implementation), applied to ambiguous words found in email text.
"""
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize

# Words that are genuinely ambiguous in business-email context — worth demoing
AMBIGUOUS_WORDS = ["interest", "charge", "close", "rate", "bank", "account", "settlement"]

POS_MAP = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}


def disambiguate(sentence: str, target_word: str):
    tokens = word_tokenize(sentence)
    if target_word not in tokens:
        return None
    tagged = pos_tag(tokens)
    tag = next((t for w, t in tagged if w == target_word), None)
    wn_pos = POS_MAP.get(tag[0]) if tag else None

    sense = lesk(tokens, target_word, pos=wn_pos)
    if sense is None:
        return {"word": target_word, "synset": None, "definition": "no sense found"}
    return {"word": target_word, "synset": sense.name(), "definition": sense.definition()}


def all_senses(word: str, pos=None):
    return [(s.name(), s.definition()) for s in wn.synsets(word, pos=pos)]


def demo_on_emails(sentences, words=AMBIGUOUS_WORDS, max_examples=3):
    results = []
    for word in words:
        found = 0
        for sent_tokens in sentences:
            sent = " ".join(sent_tokens)
            if word in sent_tokens and found < max_examples:
                result = disambiguate(sent, word)
                if result:
                    result["sentence"] = sent[:100]
                    results.append(result)
                    found += 1
            if found >= max_examples:
                break
    return results