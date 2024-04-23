from model.randomforest import RandomForest
from modelling.chainer import Chainer

def model_predict(data, chainer:Chainer=None):
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data, chainer)
    model.print_results(data, chainer,1)
    model.print_results(data, chainer,2)
