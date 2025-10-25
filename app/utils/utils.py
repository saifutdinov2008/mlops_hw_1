import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import mlflow


def validate_model(model,):

    """Simple validation: mean cosine similarity among vectors."""
    
    vectors = model.dv.vectors
    cos_sim = cosine_similarity(vectors)
    avg_sim = np.mean(cos_sim)
    return avg_sim



def plot_cosine_heatmap(model, save_path="cosine_heatmap.png"):
    """Plot heatmap of cosine similarity between vectors."""
    vectors = model.dv.vectors
    cos_sim = cosine_similarity(vectors)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cos_sim[:50, :50], cmap="viridis")  # show first 50 for clarity
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Products")
    plt.ylabel("Products")
    plt.tight_layout()
    plt.savefig(save_path)
    mlflow.log_artifact(save_path)
    plt.close()

