from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from pnet.bernoullimm import BernoulliMM
from sklearn.utils.extmath import logsumexp

@Layer.register('hyper-mixture-classification-layer')
class HyperMixtureClassificationLayer(SupervisedLayer):
    def __init__(self, n_components=1, min_prob=0.0001, block_size=0, mixture_type = "bernoulli", settings={}):
        self._n_components = n_components
        self._min_prob = min_prob
        self._models = None
        self._block_size = block_size
        self._mixture_type = mixture_type
        self._modelinstance = None
        self._settings = settings
    @property
    def trained(self):
        return self._models is not None

    @property
    def classifier(self):
        return True
        
    def score(self, X_all):
        scoreList = []
        for i in range(len(self._modelinstance)):
            Yhat = np.zeros((X_all.shape[0], X_all.shape[1]))
            blockSize = 1000
            for j in range(0, X_all.shape[0], blockSize):
                blockend = min(X_all.shape[0], j + blockSize)
                for k in range(X_all.shape[1]):
                    X = X_all[j:blockend, k]
                    Yhat[j:blockend, k] = self._modelinstance[i][k].score(X.reshape(blockend - j, -1))
            scoreList.append(Yhat)
        scoreList = np.swapaxes(scoreList, 0, 1)
        return scoreList
            
    def extract(self, X_all):
        #print "mixture classification extract started"
        llh = self.score(X_all)
        print(llh.shape)
        llh_hyperparam =  np.sum(llh, axis = 2)
        Yhat = np.argmax(llh_hyperparam, axis = 1) 
        return Yhat 
    
    def train(self, X, Y, OriginalX = None):
        K = Y.max() + 1
        mm_models = []
        self._modelinstance = []
        for k in range(K):
            Xk = X[Y == k]
            #import pdb ; pdb.set_trace()
            #print(Xk.shape)
            #print(X.shape)
            #print(Y.shape)
            Xk = Xk.reshape((Xk.shape[0], Xk.shape[1], -1))
            
            from sklearn.mixture import GMM
            hyper_mm_means = []
            hyper_mm_models = []
            for hyper_component in range(Xk.shape[1]):
                mm = GMM(n_components = self._n_components, n_iter = 20, n_init = 1, random_state=0, covariance_type = self._settings.get('covariance_type', 'diag'))
                mm.fit(Xk[:,hyper_component])
                hyper_mm_means.append(mm.means_.reshape((self._n_components,)+X.shape[2:]))
                hyper_mm_models.append(mm)
            
            mm_models.append(np.vstack(hyper_mm_means))
            self._modelinstance.append(hyper_mm_models)
        self._models = np.asarray(mm_models)
        #print "mixtureclassification-layer training finished"
    
    def save_to_dict(self):
        d = {}
        d['n_components'] = self._n_components
        d['min_prob'] = self._min_prob
        d['models'] = self._models
        d['block_size'] = self._block_size
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(n_components=d['n_components'], min_prob=d['min_prob'])
        obj._models = d['models']
        if 'block_size' in d:
            obj._block_size = d['block_size']
        else:
            obj._block_size = 0
        return obj
