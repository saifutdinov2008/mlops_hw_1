import nltk
from data.load_data import load_data
from model.train import prepare_corpus, train_doc2vec_model
from model.embeddings import get_product_vector, get_similar_products

import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')


data_path = "/Users/yusufsaifutdinov/Desktop/hw_mlops/mlops_hw_1/app/amz_total_data_limited.csv"

if __name__ == "__main__":

    # Load data
    df, stop_words = load_data(data_path)

    # Create products mapper
    products_mapper = dict(zip(df['title'].str.lower(), df['model_index'].astype(int)))

    # Prepare corpus
    corpus = prepare_corpus(df, stop_words)

    # Train Doc2Vec model
    model = train_doc2vec_model(corpus)

    # Example: get similar products
    product_name = 'etguuds White USB C to USB C Cable [10ft, 2-Pack], 60W Fast Charging Type C to Type C Charger Cable for Samsung Galaxy S23 S22 S21 S20 Ultra 5G, Z Flip/Fold 4 3, Note 20, Pixel 7 6 Pro & USB-C Laptop'
    vector, product_id = get_product_vector(model, products_mapper, product_name)
    output = get_similar_products(model, vector, topn=20, df=df)

    print(output.head())
