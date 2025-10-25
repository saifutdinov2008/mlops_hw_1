import mlflow
from mlflow.tracking import MlflowClient

class HandleModel:
    def __init__(self, model_name: str, version: int = 1):
        self.model_name = model_name
        self.version = version
        self.client = MlflowClient()

    def register_model(self, run_id: str) -> None:
        """
        Register model from the artifact folder logged in MLflow.
        Assumes model is logged under artifact_path='model'.
        """
        mlflow.register_model(f"runs:/{run_id}/model", self.model_name)


    def update_meta(self, description) -> None:
        self.client.update_model_version(
            name=self.model_name,
            version=self.version,
            description=description,
        )

    def assign_alias(self, alias: str = 'staging') -> None:
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=alias,
            version=f"{self.version}"
        )

    def tag_model(self, key: str = 'env', value: str = 'staging') -> None:
        self.client.set_model_version_tag(
            name=self.model_name,
            version=f"{self.version}",
            key=key,
            value=value
        )
