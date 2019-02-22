from sklearn import datasets
from sklearn import svm
import pylab as pl
import joblib as jl

#iris = datasets.load_iris()
#digits = datasets.load_digits()

#print(digits.data)
#print(digits.target)
#print(digits.images[0])
#
#
#clf = svm.SVC(gamma=0.001, C=100.)
#clf.fit(digits.data[:-1], digits.target[:-1])
#a = clf.predict(digits.data[-1:])
#print(a)
#b = digits.images[-1]
#pl.pcolor(b)
#pl.show()




clf = svm.SVC(gamma='auto')
iris = datasets.load_iris()
X,y = iris.data, iris.target
print(X)
print(y)
clf.fit(X[:-1],y[:-1])


jl.dump(clf,'iris.joblib')


clf2 = jl.load('iris.joblib')
a = clf2.predict(X[-1:])
print(a)



