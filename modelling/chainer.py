import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Chainer:
    encoder = None
    DELIMITER = ' > '
    def __init__(self, columns_to_chain ):
        # Create an instance of OneHotEncoder
        self.encoder = OneHotEncoder(sparse_output=False)
        self.columns_to_chain = columns_to_chain

    def chain_into_one_target_var(self, df):
        """
        Chain multiple columns into one target variable
        :param df: dataframe
        :return: dataframe
        """

        # Fit the encoder and transform the specified columns
        encoded_data = self.encoder.fit_transform(df[self.columns_to_chain])

        # Combine the encoded data into one single column (e.g., as a list or tuple)
        df['y'] = [tuple(row) for row in encoded_data]
        df['y'] = df['y'].apply(lambda t: ''.join(str(int(i)) for i in t))
        return df

    def decode_unchained(self, y):

        vfunc = np.vectorize(lambda s: [np.float64(i) for i in s], otypes=[list])
        y = vfunc(y)

        # Split the combined data and reshape it into the appropriate shape for decoding
        decoded_data = self.encoder.inverse_transform([list(row) for row in y])
        concatenated_data = np.array([self.DELIMITER.join(map(str, row)) for row in decoded_data])

        # Create a DataFrame with the decoded data
        return concatenated_data

    def remove_type(self, y: np.ndarray, num_types: int = 1):
        y = pd.Series(y)
        return y.apply(lambda x: Chainer.DELIMITER.join(x.split(Chainer.DELIMITER)[:-num_types]))