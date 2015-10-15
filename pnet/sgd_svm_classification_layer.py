from __future__ import division, print_function, absolute_import
__author__ = 'jiajunshen'
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn import linear_model


@Layer.register('sgd-svm-classification-layer')
class SGDSVMClassificationLayer(SupervisedLayer):
    def __init__(self, settings={}):
        self._settings = settings
        self._svm = None

    @property
    def trained(self):
        return self._svm is not None

    @property
    def classifier(self):
        return True

    def extract(self,X):
        Xflat = X.reshape((X.shape[0], -1))
        return self._svm.predict(Xflat)


    def train(self, X, Y, OriginalX=None):
        Xflat = X.reshape((X.shape[0], -1))
        #self._svm = linear_model.SGDClassifier(loss = "hinge",penalty = 'l2', n_iter=20, shuffle=True,verbose = False,
        #                                 learning_rate = "constant", eta0 = 0.01, random_state = 0)
        self._svm = clf = linear_model.SGDClassifier(alpha=0.01, loss = "log",penalty = 'l2', n_iter=2000, shuffle=True,verbose = False,
                                         learning_rate = "optimal", eta0 = 0.0, epsilon=0.1, random_state = None, warm_start=False,
                                         power_t=0.5, l1_ratio=1.0, fit_intercept=True)
        self._svm.fit(Xflat, Y)
        print(np.mean(self._svm.predict(Xflat) == Y))


    def save_to_dict(self):
        d = {}
        d['svm'] = self._svm
        d['settings'] = self._settings
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls()
        obj._settings = d['settings']
        obj._svm = d['svm']
        return obj