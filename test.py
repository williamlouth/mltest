import numpy as np
import pandas as pd


data = []
with open('iris','r') as fp:
    data = fp.readlines()

data2 = [i.split(',') for i in data]
df = pd.DataFrame(data2,columns = ['sepal length','sepal width','petal length','petal width','class'])
print(df)





















































