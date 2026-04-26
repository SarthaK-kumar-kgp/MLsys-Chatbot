import re

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")

_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "co", "could",
    "did", "do", "does", "doing", "don", "done", "down", "during", "each",
    "few", "for", "from", "get", "got", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "ll",
    "me", "might", "more", "most", "must", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "us",
    "ve", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "yours", "yourself", "yourselves",
    "need", "want", "help", "please", "like", "got", "know",
    "im", "ive", "dont", "tell", "let", "ur", "damn", "bloody", "freaking",
    "assist", "assistance", "gonna", "gotta", "wanna",
})


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _NON_ALNUM.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def simple_stem(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("lling") and len(word) > 6:
        return word[:-4]
    if word.endswith("ling") and len(word) > 5:
        return word[:-3]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ment") and len(word) > 5:
        return word[:-4]
    if word.endswith("tion") and len(word) > 5:
        return word[:-4]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def extract_keywords(text: str) -> set[str]:
    words = normalize_text(text).split()
    return {simple_stem(w) for w in words if w not in _STOP_WORDS and len(w) > 1}
