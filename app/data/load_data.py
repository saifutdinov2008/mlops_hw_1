import pandas as pd
from nltk.corpus import stopwords

def load_data(file_path: str):
    
    df = pd.read_csv(file_path)
    df = df.reset_index().rename(columns={'index': 'model_index'})
    df['model_index'] = df['model_index'].astype(str)
    stop_words = stopwords.words('english')
    return df, stop_words
