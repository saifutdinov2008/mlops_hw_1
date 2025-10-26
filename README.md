# mlops_hw_1

# Amazon Product Similarity with Doc2Vec

Train a Doc2Vec model on Amazon product titles to find similar products, track experiments, and deploy models with MLflow. Supports general training, hyperparameter tuning with Optuna, and MLflow model registry integration.

## Setup

1. Install dependencies with Poetry:

poetry install

(Optional) Activate virtual environment:

poetry shell

2. Download required NLTK data:
python -m nltk.downloader stopwords

3. Start MLflow UI in a separate terminal:
poetry run mlflow ui --port 5000

## Run
poetry run python app/main.py


# The script will:
Load and preprocess the Amazon dataset.
Prepare the corpus for Doc2Vec.
Run Optuna hyperparameter tuning (5 trials by default).
Train the Doc2Vec model with best hyperparameters.
Validate the model and log average similarity.
Generate and log cosine similarity heatmap.
Log artifacts, model, and hyperparameters in MLflow.
Register the model as Doc2VecModel and update metadata, alias, and tags.

# MLflow
Tracking URI: http://127.0.0.1:5000
Experiment name: doc2vec_experiment
Logged artifacts: cosine heatmap, Optuna tuning results
Registered model: Doc2VecModel

