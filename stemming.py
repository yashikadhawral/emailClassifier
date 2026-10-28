import re
from nltk.stem import PorterStemmer, SnowballStemmer

# Common email-domain words demonstrating inflectional variations
EMAIL_VOCAB_EXAMPLES = [
    "prioritized",
    "prioritizing",
    "priority",
    "deliverables",
    "delivering",
    "rescheduled",
    "scheduling",
    "reviewed",
    "reviewing",
    "urgent",
    "urgently",
    "attached",
    "attachment",
    "meeting",
    "meetings",
]


def stem_word(word: str, stemmer: PorterStemmer = None) -> str:
    """Stem a single word using PorterStemmer."""
    ps = stemmer or PorterStemmer()
    return ps.stem(word.strip())


def stem_words(words: list[str]) -> dict[str, str]:
    """Stem a list of words, returning a mapping of {original: stemmed}."""
    ps = PorterStemmer()
    return {w: ps.stem(w) for w in words}


def stem_sentence(sentence: str) -> list[tuple[str, str]]:
    """
    Tokenizes a sentence by word boundary and returns pairs of (original, stem).
    Uses regex boundary matching to avoid external tokenizer download requirements.
    """
    ps = PorterStemmer()
    tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", sentence)
    return [(tok, ps.stem(tok)) for tok in tokens]


def compare_stemmers(words: list[str] = EMAIL_VOCAB_EXAMPLES):
    """Compare PorterStemmer vs SnowballStemmer on representative email words."""
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    results = []
    for w in words:
        results.append({
            "word": w,
            "porter": porter.stem(w),
            "snowball": snowball.stem(w)
        })
    return results


def run_demo():
    print("=" * 65)
    print("STEMMING DEMONSTRATION (NLTK PorterStemmer)")
    print("=" * 65)
    print("\n1. Stemming sample email vocabulary:")
    print(f"{'Original Word':<16} | {'Porter Stem':<14} | {'Snowball Stem':<14}")
    print("-" * 50)
    for row in compare_stemmers(EMAIL_VOCAB_EXAMPLES[:8]):
        print(f"{row['word']:<16} | {row['porter']:<14} | {row['snowball']:<14}")

    sample_email_line = "Hi Yashika, we are prioritizing and rescheduling your urgent project deliverables."
    print("\n2. Stemming an email sentence:")
    print(f"Input: \"{sample_email_line}\"\n")
    pairs = stem_sentence(sample_email_line)
    stemmed_sentence = " ".join(stem for _, stem in pairs)
    print(f"Stemmed: \"{stemmed_sentence}\"")

    print("\n3. Why our deep learning pipeline uses subwords instead:")
    print("   - PorterStemmer:   'prioritizing' -> 'priorit'  (heuristic truncation)")
    print("   - WordPiece/DistilBERT: ['prior', '##iti', '##zing'] (preserves morphemic embeddings)")
    print("   - ROUGE Scorer:    Stems internally (use_stemmer=True) to match n-gram overlap.")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()

