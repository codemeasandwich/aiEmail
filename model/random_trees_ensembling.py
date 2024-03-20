import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

class RandomTreesEmbedding(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomTreesEmbedding, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10)
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions))


    def get_proba(self, X_test) -> pd.DataFrame:
        p_result = pd.DataFrame(self.predict_proba(X_test))
        p_result.columns = self.classes_
        print(p_result)
        return p_result


    def data_transform(self) -> None:
        ...

