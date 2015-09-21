from sklearn import svm, datasets
import pylab as pl

# task is to predict, given an image, which it represents

# sklearn dataset digits
digits = datasets.load_digits()

# estimator example
clf = svm.SVC(gamma=0.001, C=100.)
#better way to find gamma and C is to use grid search and cross validation

#pass training set to the fit method
clf.fit(digits.data[:-1], digits.target[:-1])

#Now predict new values, ask classifier what is the digit of out last imagein the digits dataset 
# (not in the training set)

clf.predict(digits.data[-1])
    
# show image trying to predit
pl.imshow(digits.images[-1], cmap=pl.cm.gray_r)