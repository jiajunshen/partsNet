__author__ = 'jiajunshen'

from pnet.layer import Layer
import numpy as np
from sklearn.decomposition import PCA
import amitgroup as ag

@Layer.register('pca-layer')
class PCALayer(Layer):
    def __init__(self, numOfComponents, settings={}):
        self._numOfComponents = numOfComponents
        self._settings = settings
        self._pca = None

    @property
    def trained(self):
        return self._pca is not None

    def train(self, X_tuple, Y = None, OriginalX = None):
        X, num_of_previousFeatures = X_tuple
        total_dimension = np.product(X.shape)
        flatten_x = X.reshape((total_dimension // X.shape[-1], X.shape[-1]))
        self._pca = PCA(n_components=self._numOfComponents)
        self._pca.fit(flatten_x)

    def extract(self,X_tuple, Y = None, test_accuracy = False):
        X, num_of_previousFeatures = X_tuple
        total_dimension = np.product(X.shape)
        flatten_x = X.reshape((total_dimension // X.shape[-1], X.shape[-1]))
        self._pca.transform(flatten_x)
        X = flatten_x.reshape(X.shape)
        return (X, self._numOfComponents)


    def save_to_dict(self):
        d = {}
        d['pca'] = self._pca
        d['numofComponents'] = self._numOfComponents
        d['settings'] = self._settings

        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['numofComponents'], d['settings'])
        obj._pca = d['pca']
        return obj