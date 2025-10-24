import re
from string import punctuation
from nltk.corpus import stopwords
from pymystem3 import Mystem

# Initialize lemmatizer
mystem = Mystem()

def word_tokenize_clean(doc: str, stop_words: list):
    """Tokenize string to list of words, lowercase, lemmatize, remove punctuation and stopwords."""
    tokens = list(set(mystem.lemmatize(doc.lower())))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def clean_text(text: str):
    """Remove digits and special characters."""
    return re.sub(r'[!/()0-9]', '', str(text))
