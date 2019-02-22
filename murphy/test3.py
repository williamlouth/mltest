import numpy as np
import sklearn as sk
import pylab as pl
import pandas as pd
from tempfile import TemporaryFile
import os

final_df = pd.DataFrame()

listOfFiles = os.listdir('test/')
for m in listOfFiles:
#for m in listOfFiles[0:5]:

    with open('test/'+m,'r') as fp:
        data = fp.readlines()
    
    data2 = [i.split(',') for i in data]
    
    
    df = pd.DataFrame()
    
    for j in data2:
        for i in j:
            line =i.split(' ') 
            line2 = [i  for i in line if i != '' and i != 'line']
            df2 = pd.DataFrame([line2], columns = ['char','line no','x1','y1','x2','y2','line seg l','diag size'])
            df2['x1'] = df2['x1'].astype('float')
            df2['x2'] = df2['x2'].astype('float')
            df2['y1'] = df2['y1'].astype('float')
            df2['y2'] = df2['y2'].astype('float')
            df = df.append(df2)
    
    #print(df)
    pl.figure(0)
    #pl.axis([0,20,0,20])
    
    image_array = np.zeros((8,12))
    
    imag_maxx = df[['x1','x2']].values.max()
    imag_minx = df[['x1','x2']].values.min()
    
    
    imag_maxy = df[['y1','y2']].values.max()
    imag_miny = df[['y1','y2']].values.min()
    
    imag_xrange = imag_maxx - imag_minx
    imag_yrange = imag_maxy - imag_miny
    imag_xrange +=0.1
    imag_yrange +=0.1
            
    for i in range(0,df.shape[0]):
    #for i in range(0,2):
        a = df.iloc[i]
        xs =[a['x1'],a['x2']]
        ys =[a['y1'],a['y2']]
        devitions = 10
        for i in np.arange(1/devitions,1,1/devitions):
            x = (xs[0]*i+xs[1]*(1-i))-imag_minx
            y = (ys[0]*i+ys[1]*(1-i)) - imag_miny
            x2 = int((x*8)//imag_xrange)
            y2 = int((y*12)//imag_yrange)
            image_array[x2][y2] = 1
    
        pl.plot(xs,ys)
    #print(image_array)
    
    #pl.show()
    name = m
    df3 = pd.Series(image_array.ravel())
    df3 = df3.rename(name)
    final_df = final_df.append(df3)
    #of=open("learn_raw/"+m+'.npy',"wb")
    ##of = TemporaryFile()
    #np.save(of,image_array)
    #of.close()

print(final_df)
final_df.to_hdf('test.h5',key='test')
