import random

class BarebonesKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions
        

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
clf = BarebonesKNN()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))