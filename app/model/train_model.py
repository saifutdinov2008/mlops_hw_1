from gensim.models import Doc2Vec

def train_model(corpus, vector_size=50, alpha=0.02, min_alpha=0.00025, min_count=4, epochs=20):

    """Train Doc2Vec model and return it."""
    
    model = Doc2Vec(
        vector_size=vector_size,
        alpha=alpha,
        min_alpha=min_alpha,
        min_count=min_count,
        dm=0
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)
    return model
