import mlflow

# --- PyFunc wrapper for Doc2Vec ---
class Doc2VecWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        """
        model_input: list of tokenized documents
        Returns inferred vectors for each document
        """
        return [self.model.infer_vector(doc) for doc in model_input]