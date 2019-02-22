import sklearn as sk
from sklearn import svm
import numpy as np
import pylab as pl
import pandas as pd
import os
from joblib import dump, load


    
data = pd.read_hdf('learn.h5','learn')


target = data.index.tolist()
target2 = [i[0] for i in target]


clf = svm.SVC(gamma=0.001,C=1000.)
clf.fit(data,target2)

dump(clf,'learn.joblib')


print("done")
