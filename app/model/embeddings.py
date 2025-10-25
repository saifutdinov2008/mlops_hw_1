import pandas as pd

def get_product_vector(model, products_mapper, product_name: str):
    
    """Get vector embedding of a product by name."""
    product_id = products_mapper[product_name.lower()]
    return model.dv.vectors[product_id], product_id

def get_similar_products(model, product_vector, df, topn=20):
    """
    Return top-N similar products as a dataframe with:
    - model_index
    - product_name
    - model_score
    """
    similars = model.dv.most_similar(positive=[product_vector], topn=topn)
    
    # Convert to DataFrame
    output = pd.DataFrame(similars, columns=['model_index', 'model_score'])
    
    # Convert model_index to string to match df
    output['model_index'] = output['model_index'].astype(str)
    
    # Merge with original df to get product name
    output = output.merge(df[['model_index', 'title']], on='model_index', how='left')
    output = output.rename(columns={'title': 'product_name'})
    
    # Reorder columns
    output = output[['model_index', 'product_name', 'model_score']]
    
    return output


