import optuna
import mlflow
from model.train_model import train_model

def objective(trial, corpus):
    """Optuna objective for Doc2Vec."""
    vector_size = trial.suggest_int("vector_size", 50, 200)
    alpha = trial.suggest_float("alpha", 0.01, 0.1, log=True)
    min_alpha = trial.suggest_float("min_alpha", 0.0001, 0.01, log=True)
    min_count = trial.suggest_int("min_count", 2, 10)
    epochs = trial.suggest_int("epochs", 10, 40)

    model = train_model(corpus, vector_size, alpha, min_alpha, min_count, epochs)
    
    # Simple metric: average max cosine similarity among vectors
    score = model.dv.vectors.shape[0]  # placeholder, replace with real metric if needed
    return score

def run_optuna_tuning(corpus, n_trials=5):
    """Run Optuna hyperparameter optimization with MLflow logging."""
    with mlflow.start_run(run_name="optuna_doc2vec"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, corpus), n_trials=n_trials)

        # Log best params and metric
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_value", study.best_value)

        # Save visualizations
        fig_hist = optuna.visualization.plot_optimization_history(study)
        fig_hist.write_html("optuna_history.html")
        mlflow.log_artifact("optuna_history.html")

        fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
        fig_parallel.write_html("optuna_parallel.html")
        mlflow.log_artifact("optuna_parallel.html")

    return study


