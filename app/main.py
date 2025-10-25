import warnings
import nltk
import mlflow
import mlflow.pyfunc
from preprocessing.prepare_data import load_data
from preprocessing.feature_engineering import prepare_corpus
from model.train_model import train_model
from model.optuna_tuning import run_optuna_tuning
from utils.utils import validate_model, plot_cosine_heatmap
from utils.wrapper import Doc2VecWrapper
from deployment_mlflow import HandleModel  # import your class

warnings.filterwarnings('ignore')
nltk.download('stopwords')

# MLflow setup
EXPERIMENT_NAME = "doc2vec_experiment"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

# Data config
data_path = "/Users/yusufsaifutdinov/Desktop/hw_mlops/mlops_hw_1/app/data/amz_total_data_limited.csv"

if __name__ == "__main__":
    
    # Load and preprocess
    df, stop_words = load_data(data_path)
    corpus = prepare_corpus(df, stop_words)
    
    # Run hyperparameter optimization
    study = run_optuna_tuning(corpus, n_trials=5)
    best_params = study.best_params
    
    # Optuna artifacts
    optuna_history = "optuna_history.html"
    optuna_parallel = "optuna_parallel.html"

    with mlflow.start_run(run_name="doc2vec_final_training") as run:
        # Log best hyperparameters
        mlflow.log_params(best_params)
        # Train model
        model = train_model(corpus, **best_params)

        # Validate model
        avg_similarity = validate_model(model)
        mlflow.log_metric("avg_similarity", avg_similarity)

        # Plot and log cosine similarity heatmap
        plot_cosine_heatmap(model, save_path="cosine_heatmap.png")

        # Log Optuna artifacts
        mlflow.log_artifact(optuna_history)
        mlflow.log_artifact(optuna_parallel)

        # Log and register model with PyFunc wrapper
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=Doc2VecWrapper(model),
            registered_model_name="Doc2VecModel"
        )

        # Use HandleModel to set description, alias, and tag
        handle_model = HandleModel(model_name="Doc2VecModel", version=1)
        handle_model.update_meta(description="Final Doc2Vec model trained on Amazon dataset using Optuna hyperparameters")
        handle_model.assign_alias(alias="staging")
        handle_model.tag_model(key="env", value="staging")

        # Add experiment tags
        mlflow.set_tag("model_type", "Doc2Vec")
        mlflow.set_tag("dataset", "Amazon Reviews Limited")
        mlflow.set_tag("author", "Yusuf Saifutdinov")
        mlflow.set_tag("notes", "Hyperparameter tuning with Optuna, final model validated and registered.")

        print(f"Run {run.info.run_id} complete.")
        print("Model logged, registered, and updated in MLflow Registry as 'Doc2VecModel'.")
