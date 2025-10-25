import pandas as pd
from nltk.corpus import stopwords
import re
from pymystem3 import Mystem

mystem = Mystem()

def load_data(file_path: str):

    """Load CSV and return dataframe and stopwords."""

    df = pd.read_csv(file_path)
    df = df.reset_index().rename(columns={'index': 'model_index'})
    df['model_index'] = df['model_index'].astype(str)
    stop_words = stopwords.words('english')
    return df, stop_words

def clean_text(text: str):

    """Remove digits and special characters."""

    return re.sub(r'[!/()0-9]', '', str(text))

def word_tokenize_clean(doc: str, stop_words: list):

    """Tokenize and clean text."""
    
    tokens = list(set(mystem.lemmatize(doc.lower())))
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

