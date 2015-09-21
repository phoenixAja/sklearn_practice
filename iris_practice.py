#Testing Sklearn for practice

import numpy as np
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier

def load_iris_data():
    #load the iris dataset
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_Y = iris.target
    return iris_X, iris_Y

def split_data(X, Y):
    # Split iris data in train and test data sets
    np.random.seed(0)
    indeces = np.random.permutation(len(X))
    X_train = X[indeces[:-10]]
    Y_train = Y[indeces[:-10]]
    X_test = X[indeces[-10:]]
    Y_test = Y[indeces[-10:]]
    return X_train, Y_train, X_test, Y_test

def create_classifier(X_train, Y_train, X_test):
    # Make our model to predict Y values
    knn = KNeighborsClassifier()
    classifier = knn.fit(X_train, Y_train)
    prediction = knn.predict(X_test)
    return classifier, prediction

def logistic_regression(x_train, y_train):
    # fit for sigmoid function (logistic)
    logistic = linear_model.LogisticRegression(C=1e5)
    log_fit = logistic.fit(x_train, y_train)
    return(log_fit)

def main():
    iris_X, iris_Y = load_iris_data()
    #get unique iris types (classifier)
    np.unique(iris_Y)
    iris_x_train, iris_y_train, iris_x_test, iris_y_test = split_data(iris_X, iris_Y)
    classifier, prediction = create_classifier(iris_x_train, iris_y_train, iris_x_test)
     
    #Results
    print prediction
    print iris_y_test
    logistic_regression(iris_x_train, iris_y_train)

if __name__ == "__main__":
    main()