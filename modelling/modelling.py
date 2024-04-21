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
    
    return results
def model_predict_chained(data, df, name):
    results = []
    
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_y2())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
    
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_y2_y3())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
    
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_y2_y3_y4())
    model.train(data)
    model.predict(data.X_test)
    res = model.print_results(data)
    results.append(res)
    
    return results
def model_predict_hierarchical(data, df, name):
    results = []
    
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_y2())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
    
    y2_classes = np.unique(data.get_y2())
    for y2_class in y2_classes:
        filtered_data = data.filter_by_y2(y2_class)
        
        model = RandomForest(f"RandomForest_y2_{y2_class}", filtered_data.get_embeddings(), filtered_data.get_y3())
        model.train(filtered_data)
        model.predict(filtered_data.X_test)
        model.print_results(filtered_data)
        
        y3_classes = np.unique(filtered_data.get_y3())
        for y3_class in y3_classes:
            filtered_data_y3 = filtered_data.filter_by_y3(y3_class)
            
            model = RandomForest(f"RandomForest_y2_{y2_class}_y3_{y3_class}", filtered_data_y3.get_embeddings(), filtered_data_y3.get_y4())
            model.train(filtered_data_y3)
            model.predict(filtered_data_y3.X_test)
            res = model.print_results(filtered_data_y3)
            results.append(res)
    
    return results
def model_evaluate(model, data):
    model.print_results(data)