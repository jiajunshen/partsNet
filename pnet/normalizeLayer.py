
from __future__ import division, print_function, absolute_import
__author__ = 'jiajunshen'
import amitgroup as ag
import numpy as np
from pnet.layer import Layer

@Layer.register('normalize-layer')
class NormalizeLayer(Layer):
    def __init__(self):
        self._mean = None


    @property
    def trained(self):
        return self._mean is not None

    def train(self, X, Y = None, OriginalX = None):
        self._mean = np.mean(X, axis = 0)

    def extract(self, X):
        #return (X - self._mean).astype(np.float32)
        return (X).astype(np.float32)

    def save_to_dict(self):
        d = {}
        d['mean'] =  self._mean
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls()
        obj._mean = d['mean']
        return obj

