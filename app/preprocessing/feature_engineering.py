from gensim.models.doc2vec import TaggedDocument
from preprocessing.prepare_data import clean_text, word_tokenize_clean

def prepare_corpus(df, stop_words):

    """Preprocess and create corpus for Doc2Vec."""
    
    tags_corpus = df['title'].values
    tags_corpus = [clean_text(x) for x in tags_corpus]
    corpus = [
        TaggedDocument(words=word_tokenize_clean(D, stop_words), tags=[str(i)])
        for i, D in enumerate(tags_corpus)
    ]
    return corpus
