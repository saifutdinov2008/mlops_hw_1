from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocessing.prepare_data import word_tokenize_clean, clean_text

def prepare_corpus(df, stop_words):
    """Preprocess titles and create TaggedDocument corpus."""
    tags_corpus = df['title'].values
    tags_corpus = [clean_text(x) for x in tags_corpus]
    corpus = [TaggedDocument(words=word_tokenize_clean(D, stop_words), tags=[str(i)])
              for i, D in enumerate(tags_corpus)]
    return corpus

def train_doc2vec_model(corpus, vector_size=50, alpha=0.02, min_alpha=0.00025, min_count=4, epochs=20):
    """Train a Doc2Vec model and return it."""
    model = Doc2Vec(vector_size=vector_size, alpha=alpha, min_alpha=min_alpha, min_count=min_count, dm=0)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)
    return model
