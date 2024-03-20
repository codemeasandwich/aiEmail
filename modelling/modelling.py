from model.SGD import SGD
from model.randomforest import RandomForest
from model.adaboost import AdaBoost
from model.voting import Voting
from model.hist_gb import Hist_GB
from model.random_trees_ensembling import RandomTreesEmbedding


def model_predict(data, df, name):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print("Hist_GB")
    model = Hist_GB("Hist_GB", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    res = model.print_results(data)
    results.append(res)

    print("SGD")
    model = SGD("SGD", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print("AdaBoost")
    model = AdaBoost("AdaBoost", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print("Voting")
    model = Voting("Voting", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print("RandomTreesEmbedding")
    model = RandomTreesEmbedding("RandomTreesEmbedding", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)