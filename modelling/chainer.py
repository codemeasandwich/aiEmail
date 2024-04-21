import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Chainer:
    encoder = None

    def __init__(self, columns_to_chain ):
        # Create an instance of OneHotEncoder
        self.encoder = OneHotEncoder(sparse_output=False)
        self.columns_to_chain = columns_to_chain

    def chain_into_one_target_var(self, df):
        """
        Chain multiple columns into one target variable
        :param columns_to_chain: list of columns to chain
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

        vfunc = np.vectorize(lambda s: [float(i) for i in s], otypes=[list])

        y = vfunc(y)

        # Reshape y to the appropriate shape for decoding
        y = y.reshape(-1, 1)


        # Split the combined data and reshape it into the appropriate shape for decoding
        decoded_data = self.encoder.inverse_transform(y)

        # Create a DataFrame with the decoded data
        decoded_df = pd.DataFrame(decoded_data, columns=self.columns_to_chain)

        return decoded_df
