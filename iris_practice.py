import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

#get unique iris types (classifier)
np.unique(iris_Y)

# Split iris data in train and test data randomly
np.random.seed(0)
indeces = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indeces[:-10]]
iris_Y_train = iris_Y[indeces[:-10]]
iris_X_test = iris_X[indeces[-10:]]
iris_Y_test = iris_Y[indeces[-10:]]

# Make our model to predict Y values
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_Y_train)

#Now try to predict iris_Y_test
prediction = knn.predict(iris_X_test)

#Results
print prediction
print iris_Y_test
