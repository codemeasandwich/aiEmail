from modelling.chainer import  Chainer
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling(data: Data, chainer:Chainer =None):
    model_predict(data, chainer)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        chained_cols = ['y2', 'y3', 'y4']
        print("Name:", name, "Chained Columns: ", chained_cols)
        chainer = Chainer(chained_cols)
        X, group_df = get_embeddings(group_df)
        group_df = chainer.chain_into_one_target_var(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, chainer)