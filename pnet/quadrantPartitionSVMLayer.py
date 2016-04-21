__author__ = 'jiajunshen'
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
from sklearn import linear_model
from multiprocessing import Pool, Value, Array
from sklearn.utils.extmath import (safe_sparse_dot, logsumexp, squared_norm)

shared_data = None

def init(_data):
    global shared_data
    shared_data = _data


shared_X = None
shared_Y = None

def init_Image(X, Y):
    global shared_X
    global shared_Y
    shared_X = X
    shared_Y = Y

@Layer.register('quadrant-partition-svm-layer')
class QuadrantPartitionSVMLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        # settings: outer_frame, random_seed,samples_per_image,patch_extraction_seed,threshold
        #, outer_frame=1, threshold=1):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._settings = settings
        self._train_info = {}
        self._partition = []
        self._parts = None
        self._svms = None

    @property
    def trained(self):
        return self._parts is not None

    def train(self, X, Y, OriginalX = None):
        return self.split_train_svms(X, Y)

    def split_train_svms(self, X, Y):
        ag.info("Split training SVMs")
        import time,random
        num_of_processor = 6
        class1, class2 = split_2vs2(10)
        round_each_proc = self._num_parts // 4 //num_of_processor
        rs = np.random.RandomState(self._settings.get('random_seeds', 0))
        #four quadrants

        p = Pool(num_of_processor, initializer=init_Image, initargs=(X,Y,))
        allObjects = []
        for i in range(4):
            args = [(i, j, round_each_proc, self._part_shape, (3,3), class1[round_each_proc * j: round_each_proc * (j + 1)],
                 class2[round_each_proc * j: round_each_proc * (j + 1)], self._settings.get('random_seeds', 0)) for j in range(num_of_processor)]

            svm_objects = p.map(task_spread_svms, args)
            for objects in svm_objects:
                allObjects+=objects

        self._svms = allObjects
        self._parts = []
        self._coef = []
        self._intercept = []
        for i in range(self._num_parts):
            print(i, self._num_parts)
            svm_coef = allObjects[i].coef_
            svm_intercept = allObjects[i].intercept_
            allParam = np.zeros(svm_coef.shape[1] + 1)
            allParam[:svm_coef.shape[1]] = svm_coef[0,:]
            allParam[-1] = svm_intercept
            std = np.std(allParam) * 20
            #std = 1
            print(np.std(allParam/std),np.max(allParam/std))
            self._parts.append((svm_coef/std, svm_intercept/std))
            self._coef.append(svm_coef/std)
            self._intercept.append(svm_intercept/std)

        self._coef = np.vstack(self._coef)
        self._intercept = np.vstack(self._intercept)
        print(np.max(self._coef))
        print(np.mean(self._coef))
        print(np.max(self._intercept))
        print(np.mean(self._intercept))

    def extract(self,X_all, Y = None, test_accuracy = False):
        ag.info("randomPartition SVM start extracting")
        outer_frame = self._settings.get('outer_frame', 0)
        dim = (X_all.shape[1],X_all.shape[2])
        if  outer_frame != 0:
            XX = np.zeros((X_all.shape[0], X_all.shape[1] + 2 * outer_frame, X_all.shape[2] + 2 * outer_frame, X_all.shape[3]), dtype=np.float16)
            XX[:, outer_frame:X_all.shape[1] + outer_frame, outer_frame:X_all.shape[2] + outer_frame,:] = X_all
            X_all = XX
        numcl = 2
        if(numcl == 2):
            feature_map = np.zeros((X_all.shape[0],) + dim + (self._num_parts,),dtype=np.float16)
        else:
            feature_map = np.zeros((X_all.shape[0],) + dim + (numcl, self._num_parts,),dtype=np.float16)
        print("Before blowing up the memory")

        roundNumber = 100
        numEachRound = X_all.shape[0] // roundNumber
        for round in range(roundNumber):
            print(round)
            X = np.array(X_all[numEachRound * round : numEachRound * (round + 1)], dtype = np.float16)
            X_num = X.shape[0]
            import itertools
            argList = list(itertools.product(range(dim[0]),range(dim[1])))
            p = Pool(4)
            args = ((x, y, self._coef,self._intercept,
                     X[:,x:x+self._part_shape[0],
                     y:y+self._part_shape[1],:].reshape(X_num,-1).astype(np.float16)) for(x,y) in argList
                    )
            count = 0
            for x, y, score in p.imap(calcScore, args):
                feature_map[numEachRound * round : numEachRound * (round + 1),x, y,:] = score
                count+=1
            p.terminate()

        print(feature_map[0,0,0,0])
        return (feature_map,self._num_parts)

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['training_info'] = self._train_info
        d['partition'] = self._partition
        d['parts'] = self._parts
        d['svms'] = self._svms
        d['coef'] = self._coef
        d['intercept'] = self._intercept
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'],d['part_shape'],d['settings'])
        obj._train_info = d['training_info']
        obj._partition = d['partition']
        obj._parts = d['parts']
        obj._svms = d['svms']
        obj._coef = d['coef']
        obj._intercept = d['intercept']
        return obj


def calcScore((x, y, svmCoef, svmIntercept, data)):
    result = (np.dot(data, svmCoef.T) + svmIntercept.T).astype(np.float16)
    if np.any(np.isnan(result)) and (not np.all(np.isfinite(result))):
        print ("result is nan")
        print(result)
        print(data)
        print(svmCoef.T)
        print(svmIntercept.T)

    return x, y, result

def task_spread_svms((quad, i, round_each_proc, part_shape, sample_shape, class1, class2, currentSeeds)):
    svm_objects = []
    for j in range(round_each_proc * i, round_each_proc * (i + 1)):
        #everytime we pick only one location,
        #print j
        quadIndex_x = quad % 2
        x_size = shared_X.shape[1] // 2
        quadIndex_y = quad // 2
        y_size = shared_X.shape[2] // 2
        patches, patches_label = get_color_patches_location(shared_X[:, quadIndex_x * x_size:(quadIndex_x + 1) * x_size,
                                                            quadIndex_y * y_size : (quadIndex_y + 1) * y_size, :],
                                                            shared_Y, class1[j - i * round_each_proc], class2[j - i * round_each_proc],
                                                            locations_per_try=1, part_shape=part_shape,
                                                            sample_shape=sample_shape,fr = 1, randomseed=j + currentSeeds,
                                                            threshold=0, max_samples=300000)

        clf = linear_model.SGDClassifier(alpha=0.001, loss = "log",penalty = 'l2', n_iter=20, shuffle=True,verbose = False,
                                         learning_rate = "optimal", eta0 = 0.0, epsilon=0.1, random_state = None, warm_start=False,
                                         power_t=0.5, l1_ratio=1.0, fit_intercept=True)
        clf.fit(patches, patches_label)
        print(np.mean(clf.predict(patches) == patches_label))
        svm_objects.append(clf)
    return svm_objects

def get_color_patches_location(X, Y, class1, class2, locations_per_try, part_shape, sample_shape, fr, randomseed, threshold, max_samples):
    assert X.ndim == 4
    channel = X.shape[-1]
    patches = []
    patches_label = []
    rs = np.random.RandomState(randomseed)

    #th = self._settings['threshold']
    th = threshold
    w, h = [X.shape[1 + j]-part_shape[j] - sample_shape[j] +2 for j in range(2)]
    #print w,h
    indices = list(itr.product(range(w-1), range(h-1)))
    rs.shuffle(indices)
    i_iter = itr.cycle(iter(indices))
    count = 0
    for trie in range(locations_per_try):
        x, y = next(i_iter)
        #print "sampled_x"
        #print(x,y)
        for i in range(X.shape[0]):
            Xi = X[i]
            # How many patches could we extract?
            if(Y[i] not in class1 and Y[i] not in class2):
                continue
            count+=1
            for x_i in range(sample_shape[0]):
                for y_i in range(sample_shape[1]):
                    #print(x_i, y_i)
                    selection = [slice(x + x_i, x + x_i + part_shape[0]), slice(y + y_i, y+ y_i + part_shape[1])]
                    patch = Xi[selection]
                    #edgepatch_nospread = edges_nospread[selection]
                    if fr == 0:
                        tot = patch.sum()
                    else:
                        tot = abs(patch[fr:-fr,fr:-fr]).sum()

                    if th <= tot * channel:
                        patches.append(patch)
                        if(Y[i] in class1):
                            patches_label.append(0)
                        else:
                            patches_label.append(1)
                        if len(patches) >= max_samples:
                            patches = np.asarray(patches)
                            patches = patches.reshape((patches.shape[0], -1))
                            return np.asarray(patches),np.asarray(patches_label).astype(np.uint8)

    #ag.info('ENDING: {} patches'.format(len(patches)))
    patches = np.asarray(patches)
    patches = patches.reshape((patches.shape[0], -1))
    return np.asarray(patches),np.asarray(patches_label).astype(np.uint8)

def split_2vs2(n):
    result = [[],[]]
    path = []
    findSets(result, path, n, 0)
    return result[0], result[1]

def findSets(result, path, n, currentIndex):
    import copy
    if (len(path) == 4):
        print "===="
        print(len(path))
        print(path)
        print(len(result))
        currentResult = [[],[]]
        currentResult[0] = [path[0], path[0]]
        for i in range(1, 4):
            currentResult[0][1] = path[i]
            currentResult[1] = [k for k in path[1:4] if k!=path[i]]
            print(currentResult)
            result[0].append(currentResult[0])
            result[1].append(currentResult[1])
    elif (currentIndex == n):
        return
    else:
        for i in range(currentIndex, n):
            newpath = copy.deepcopy(path)
            newpath.append(i)
            findSets(result, newpath, n, i + 1)



