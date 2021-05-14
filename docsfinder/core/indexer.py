import re
from typing import Dict, Iterable, Match, Pattern

from nltk.stem.snowball import SnowballStemmer
from spacy import load


class Indexer:
    def __init__(self):
        self.nlp = load("en_core_web_sm")
        self.stemmer = SnowballStemmer(language="english")

    def get_indexes(self, text: str, remove_stopwords: bool = True) -> Iterable[str]:
        lower_text = text.lower()
        expanded_text = expand_contractions(lower_text)
        cleaned_text = clean_text(expanded_text)
        preprocessed_text = remove_extra_spaces(cleaned_text)
        doc = self.nlp(preprocessed_text)
        result = set()
        for token in doc:
            if token_is_valid(token, remove_stopwords):
                temp = self.stemmer.stem(token.text)
                result.add(temp)
        return result


def remove_extra_spaces(text: str) -> str:
    return re.sub(" +", " ", text)


def clean_text(text: str) -> str:
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub("\n", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^a-z]", " ", text)
    return text


def token_is_valid(token, remove_stopwords: bool) -> bool:
    if remove_stopwords and token.is_stop:
        # print(f"Token '{token}' is a stopword.")
        return False
    if token.is_punct:
        # print(f"Token '{token}' is a punct.")
        return False
    if token.is_space:
        # print(f"Token '{token}' is a space.")
        return False
    if token.pos_ in ["ADP", "AUX", "CONJ", "DET", "PART", "PRON", "SCONJ"]:
        return False
    return True


def expand_contractions(text: str) -> str:
    def repl(match: Match[str]) -> str:
        return contractions_dict[match.group(0)]

    return contractions_re.sub(repl, text)


# Dictionary of english Contractions
contractions_dict: Dict[str, str] = {
    "ain't": "are not",
    "'s": " is",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "that'd": "that would",
    "that'd've": "that would have",
    "there'd": "there would",
    "there'd've": "there would have",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what've": "what have",
    "when've": "when have",
    "where'd": "where did",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who've": "who have",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

# Regular expression for finding contractions
contractions_re: Pattern[str] = re.compile("(%s)" % "|".join(contractions_dict.keys()))
