from numpy import asarray
import numpy as np
X,y=list(),list()

z=[1,2,3,4,5,6]
X.extend(z)

y=[10,11,12,13,14,15]
X.extend(y)

X=asarray(X)
X=X.astype('float32')
print(X)
from numpy import load

data=load('5-celebrity-faces-dataset.npz')
trainX,trainy,testX,testy=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

print('Loaded: ',trainX.shape,trainy.shape,testX.shape,testy.shape)