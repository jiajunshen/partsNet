__author__ = 'jiajunshen'

from pnet.layer import Layer
from sklearn.svm import LinearSVC
from pnet.layer import SupervisedLayer
from sklearn import cross_validation
import numpy as np
import amitgroup as ag

@Layer.register('pca-layer')
class IntermediateSupervisionLayer(Layer):
    def __init__(self, numOfFeatures, penalty = 1,  settings={}):
        self._numOfFeatures = numOfFeatures
        self._settings = settings
        self._selectedFeatureIndex = None
        self._penalty = penalty
        self._svm = None

    @property
    def trained(self):
        return self._selectedFeatureIndex is not None

    @property
    def classifier(self):
        return True

    def train(self, X_tuple, Y, OriginalX = None):
        X, numFeatures = X_tuple
        assert self._numOfFeatures <= numFeatures
        Xflat = X.reshape((X.shape[0], -1))
        clf = LinearSVC(C=self._penalty, random_state=self._settings.get('seed', 0))
        print("start fitting")
        clf.fit(Xflat, Y)
        allCoef = clf.coef_.reshape((10,) + X.shape[1:])
        allCoef = abs(allCoef)
        allCoef = np.sum(allCoef,axis = 0)
        allCoef = np.sum(allCoef,axis = 0)
        allCoef = np.sum(allCoef,axis = 0)
        print("finish fitting")
        self._selectedFeatureIndex = np.argsort(allCoef)[::-1][:self._numOfFeatures]

    def extract(self,X_tuple, Y = None, test_accuracy = False):
        X, numFeatures = X_tuple
        print("Inside extraction")
        return (X[:,:,:,self._selectedFeatureIndex], self._numOfFeatures)


    def save_to_dict(self):
        d = {}
        d['svm'] = self._svm
        d['numofFeatures'] = self._numOfFeatures
        d['settings'] = self._settings
        d['selectedFeatureIndex'] = self._selectedFeatureIndex
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['numofFeatures'], d['settings'])
        obj._selectedFeatureIndex = d['selectedFeatureIndex']
        obj._svm = d['svm']
        return obj