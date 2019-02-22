import sklearn as sk
from sklearn import svm
import numpy as np
import pylab as pl
import pandas as pd
import os
from joblib import dump, load


clf = load('learn.joblib')

data = pd.read_hdf('test.h5','test')
print(data.shape)

tot = 0
t = 0
f=0
for i in range(0,4999):
    tot +=1
    a = clf.predict(data.iloc[i].values.reshape(1,-1))
    b = data.iloc[i].name[0]
    if a == b:
        t+=1
    else:
        f+=1

print("number total " + str(tot))
print("number true " + str(t))
print("number false " + str(f))
print(t/tot)
