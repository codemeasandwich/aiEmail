from model.randomforest import RandomForest
from modelling.chainer import Chainer


def model_predict(data, chainer:Chainer=None):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test, chainer)
    model.print_results(data, chainer)


def model_evaluate(model, data):
    model.print_results(data)