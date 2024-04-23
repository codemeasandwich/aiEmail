import numpy as np
import pandas as pd

from Config import Config
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
import pickle

from modelling.chainer import Chainer

num_folds = 0
seed = 0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series) -> None:
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data, chainer: Chainer = None, remove_types: int = 0) -> None:
        y_test = data.y_test
        predictions = self.predictions
        if chainer is not None:
            y_test = chainer.decode_unchained(data.y_test)
            predictions = chainer.decode_unchained(predictions)
        if remove_types > 0:
            y_test = chainer.remove_type(y_test, remove_types)
            predictions = chainer.remove_type(predictions, remove_types)
        print(classification_report(y_test, predictions))

    def data_transform(self) -> None:
        ...
