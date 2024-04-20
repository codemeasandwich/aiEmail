from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

#make dataset
X, Y = make_multilabel_classification(n_samples=12, n_classes=5,     n_labels=3,  random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

base_lr = LogisticRegression(solver='lbfgs', random_state=0)
chain = ClassifierChain(base_lr, order='random', random_state=0)
chain.fit(X_train, Y_train).predict(X_test)

chain.predict_proba(X_test)