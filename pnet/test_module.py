__author__ = 'jiajunshen'

from pnet.layer import Layer
import numpy as np
import amitgroup as ag
import pnet
from pnet.cyfuncs import activation_map_pooling as poolf


def python_poolf(X, F, shape, strides, relu):
    n = X.shape[0]
    col = X.shape[1]
    row = X.shape[2]
    channel = X.shape[3]
    result = np.zeros((n, col//2, row//2, channel))
    for i in range(n):
        for m in range(col//2):
            for n in range(row//2):
                for c in range(channel):
                    result[i,m,n,c] = np.max(X[i,2*m:2*m + 2, 2*n:2*n + 2,c])
                    if(relu):
                        result[i,m,n,c] = max(0,result[i,m,n,c])
    return result






X = np.random.rand(10000,20,20,100).astype(np.float32)
F = 100
shape = (2,2)
strides = (2,2)

result_cython = poolf(X, F, shape, strides, False)
diffArray = result_cython - python_poolf(X, F, shape, strides, False)
diffArray = diffArray.reshape(diffArray.shape[0], -1)
print(np.mean(diffArray, axis = 0))
