import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

class RandomTreesEmbedding(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomTreesEmbedding, self).__init__(model_name, embeddings, y)
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
        print(classification_report(data.y_test, self.predictions, zero_division=1))


    def get_proba(self, X_test) -> pd.DataFrame:
        p_result = pd.DataFrame(self.predict_proba(X_test))
        p_result.columns = self.classes_
        print(p_result)
        return p_result


    def data_transform(self) -> None:
        # Check if the input data is a pandas DataFrame
        if isinstance(self.embeddings, pd.DataFrame):
            # Perform data transformation
            numeric_features = self.embeddings.select_dtypes(include=[np.number]).columns
            categorical_features = self.embeddings.select_dtypes(include=['object', 'category']).columns
            # Create a ColumnTransformer for preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])
            # Fit and transform the data
            self.embeddings = preprocessor.fit_transform(self.embeddings)
        else:
            # If the input data is not a DataFrame, assume it's already transformed
            self.embeddings = self.embeddings
        # Convert the target variable to integers if needed
        label_encoder = LabelEncoder()
        if isinstance(self.y, pd.Series):
            self.y = label_encoder.fit_transform(self.y)
        else:
            self.y = label_encoder.fit_transform(self.y)

