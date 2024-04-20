import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        y = df.y.to_numpy()
        y_series = pd.Series(y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value)<1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]


        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good)
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X


    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train

