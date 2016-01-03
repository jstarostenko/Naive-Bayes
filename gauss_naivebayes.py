from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

def gauss(X,y):
	clf = GaussianNB()
	clf.fit(X,y)
	return clf
