from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from Config import *

class BaseModel(ABC):
    def __init__(self, model_name: str, X: np.ndarray, y: np.ndarray) -> None:
        self.model_name = model_name
        common_indices = np.arange(min(X.shape[0], y.shape[0]))
        X = X[common_indices]
        y = y[common_indices]
        y_series = pd.Series(y)
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
        
        if len(good_y_value) < 1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return
        
        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]
        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good
        )
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X
        
        train_indices = np.where(np.isin(y, self.y_train))[0]
        self.X_train_subset = X[train_indices]
        
        test_indices = np.where(np.isin(y, self.y_test))[0]
        self.X_test_subset = X[test_indices]
        
    @abstractmethod
    def train(self, data) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        :param data: Data object containing the training data
        :return: None
        """
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    @abstractmethod
    def predict(self, X_test: pd.Series) -> None:
        """
        Predict the labels for the given test data.
        :param X_test: Test data
        :return: None
        """
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    @abstractmethod
    def print_results(self, data) -> None:
        """
        Print the evaluation results of the model.
        :param data: Data object containing the test data
        :return: None
        """
        print(f"Results for {self.model_name}:")
        print(classification_report(data.y_test, self.predictions, zero_division=1))
        print("Confusion Matrix:")
        print(confusion_matrix(data.y_test, self.predictions))

    def data_transform(self) -> None:
        """
        Perform data transformation on the input data.
        :return: None
        """
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
        if isinstance(self.y, pd.Series):
            self.y = self.y.astype(int)
        else:
            self.y = self.y.astype(int)

    def build(self, values=None):
        """
        Build the model with the provided values.
        :param values: Dictionary of values to update the model's attributes
        :return: The updated model instance
        """
        if values is None:
            values = {}
        elif not isinstance(values, dict):
            raise ValueError("Values must be a dictionary.")

        self.__dict__.update(values)
        return self