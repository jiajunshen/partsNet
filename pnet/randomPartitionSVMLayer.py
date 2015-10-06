__author__ = 'jiajunshen'
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
from multiprocessing import Pool, Value, Array


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


@Layer.register('raondom-partition-svm-layer')
class RandomPartitionSVMLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        # settings: outer_frame, random_seed,samples_per_image,patch_extraction_seed,threshold
        #, outer_frame=1, threshold=1):
        self._num_parts = num_parts
        self._part_shape = part_shape
        #self._outer_frame = outer_frame
        #self._threshold = threshold
        self._settings = settings
        self._train_info = {}
        self._partition = []
        self._parts = None
        self._svms = None


    #Once we have all the coefficients, we could calculate the W*x + b
    @property
    def trained(self):
        return self._parts is not None

    def train(self, X, Y, OriginalX = None):
        assert Y is not None
        ag.info('Extracting patches')
        if(self._settings.get("all_locations",True)):
            patches, patches_label, patches_original = self._get_color_patches(X,Y,OriginalX)
            ag.info('Done extracting patches')
            ag.info('Training patches', patches.shape)
            ag.info('Training patches labels', patches_label.shape)
            return self.train_from_samples(patches,patches_label, patches_original)
        else:
            ag.info('Start multiple svm training: each svm is trained using patches from a specific location and a certain partition')
            return self.split_train_svms(X,Y)

    def extract(self,X, Y = None, test_accuracy = False):
        ag.info("randomPartition SVM start extracting")
        outer_frame = self._settings.get('outer_frame', 0)
        dim = (X.shape[1],X.shape[2])
        if  outer_frame != 0:
            XX = np.zeros((X.shape[0], X.shape[1] + 2 * outer_frame, X.shape[2] + 2 * outer_frame, X.shape[3]), dtype=np.float16)
            XX[:, outer_frame:X.shape[1] + outer_frame, outer_frame:X.shape[2] + outer_frame,:] = X
            X = XX

        numcl = 2
        if(numcl == 2):
            feature_map = np.zeros((X.shape[0],) + dim + (self._num_parts,),dtype=np.float16)
        else:
            feature_map = np.zeros((X.shape[0],) + dim + (numcl, self._num_parts,),dtype=np.float16)

        print X.shape, feature_map.shape
        p = Pool(4)
        from cyfuncs import transfer_label
        #partitionLabel = transfer_label(np.array(Y).astype(np.int64), np.array(self._partition).astype(np.int64), np.int64(self._num_parts), np.int64(10))
        #predict_mean = np.zeros(self._num_parts)
        for x in range(dim[0]):
            for y in range(dim[1]):
                Xr = X[:,x:x+self._part_shape[0],y:y+self._part_shape[1],:]
                Xr_flat = Xr.reshape((X.shape[0], -1)).astype(np.float16)
                args = ((self._svms[i],Xr_flat) for i in range(self._num_parts))
                scores = p.map(calcScore, args)
                scores = np.hstack(scores)
                feature_map[:,x, y,:] = scores.astype(np.float16)
                #for i in range(self._num_parts):
                    #predict_mean[i] += np.mean(self._svms[i].predict(Xr_flat) == partitionLabel[i,:])
        #for i in range(self._num_parts):
        #    print(predict_mean[i]/(dim[0] * dim[1]))
        p.close()
        print(feature_map.shape)
        print(np.any(np.isnan(feature_map)))
        print(np.all(np.isfinite(feature_map)))
        return (feature_map,self._num_parts)


    def _get_color_patches(self, X, Y, OriginalX):
        assert X.ndim == 4
        assert OriginalX.ndim == 4
        channel = X.shape[-2]
        samples_per_image = self._settings.get('samples_per_image', 20)
        fr = self._settings['outer_frame']
        patches = []
        patches_label = []
        patches_original = []
        rs = np.random.RandomState(self._settings.get('patch_extraction_seed', 0))

        th = self._settings['threshold']
        for i in range(X.shape[0]):
            Xi = X[i]
            OriginalXi = OriginalX[i]
            # How many patches could we extract?
            w, h = [Xi.shape[j]-self._part_shape[j]+1 for j in range(2)]

            # TODO: Maybe shuffle an iterator of the indices?
            indices = list(itr.product(range(w-1), range(h-1)))
            rs.shuffle(indices)
            i_iter = itr.cycle(iter(indices))
            for sample in range(samples_per_image):
                N = 200
                for tries in range(N):
                    x, y = next(i_iter)
                    selection = [slice(x, x+self._part_shape[0]), slice(y, y+self._part_shape[1])]

                    patch = Xi[selection]
                    #edgepatch_nospread = edges_nospread[selection]
                    if fr == 0:
                        tot = patch.sum()
                    else:
                        tot = abs(patch[fr:-fr,fr:-fr]).sum()

                    if th <= tot * channel:
                        patches.append(patch)
                        patches_label.append(Y[i])
                        vispatch = OriginalXi[selection]
                        span = vispatch.min(),vispatch.max()
                        if span[1]-span[0] > 0:
                            vispatch = (vispatch - span[0])/(span[1] - span[0])
                        patches_original.append(vispatch)
                        if len(patches) >= self._settings.get('max_samples', np.inf):
                            return np.asarray(patches),np.asarray(patches_label).astype(np.uint8),np.asarray(patches_original)
                        break

                    if tries == N-1:
                        ag.info('WARNING: {} tries'.format(N))






        return np.asarray(patches),np.asarray(patches_label).astype(np.uint8),np.asarray(patches_original)
    def random_partition(self, sampleNumber = 5):
        """
        randomly partition the groups into two groups.
        :ivar
            trainingLabel: the labels of the training data
            testLabel: the labels of the testing data
        :returns
            newTrainLabel
            newTestLabel
            sampledGroup
        """
        #np.random.seed(self._settings.get('random_seed', 0))
        for i in range(self._num_parts):
            sampledGroup = np.zeros(10).astype(np.uint8)
            sampledGroup[np.random.choice(range(10),size = sampleNumber,replace=False)] = 1
            self._partition.append(sampledGroup)
        return

    def train_from_samples(self, patches, patches_label, patches_original):
        ag.info("training from samples starts")
        self.random_partition()
        import time
        p = Pool(4, initializer=init, initargs=(patches,))
        patches_label_list = [patches_label for i in range(4)]


        args = ((patches_label_list[i], self._partition, i, self._num_parts // 4, ) for i in range(4))
        start_time = time.time()
        svm_objects = p.map(task_spread, args)
        allObjects = []
        for objects in svm_objects:
            allObjects+=objects
        p.close()
        print("--- %s seconds ---" % (time.time() - start_time))
        print len(allObjects)
        self._svms = allObjects
        self._parts = []
        for i in range(self._num_parts):
            svm_coef = allObjects[i].coef_
            svm_intercept = allObjects[i].intercept_
            self._parts.append((svm_coef, svm_intercept))

    def split_train_svms(self, X, Y):
        ag.info("Split training SVMs")
        import time,random
        p = Pool(4, initializer=init_Image, initargs=(X,Y,))
        class1, class2 = split_1vs2(10)
        combined = zip(class1,class2)
        rs = np.random.RandomState(0)
        rs.shuffle(combined)
        combined = combined[:self._num_parts]
        class1[:],class2[:] = zip(*combined)
        print(len(class1), len(class2))
        round_each_proc = self._num_parts//4
        args = [(i, round_each_proc, self._part_shape, (3,3), class1[round_each_proc * i: round_each_proc * (i + 1)],
                 class2[round_each_proc * i: round_each_proc * (i + 1)]) for i in range(4)]
        start_time = time.time()
        svm_objects = p.map(task_spread_svms, args)
        #init_Image(X,Y)
        #svm_objects = task_spread_svms(args[0])
        allObjects = []
        for objects in svm_objects:
            allObjects+=objects
        p.close()
        print("--- %s seconds ---" % (time.time() - start_time))
        print len(allObjects)
        self._svms = allObjects
        self._parts = []
        for i in range(self._num_parts):
            svm_coef = allObjects[i].coef_
            svm_intercept = allObjects[i].intercept_
            self._parts.append((svm_coef, svm_intercept))





    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['training_info'] = self._train_info
        d['partition'] = self._partition
        d['parts'] = self._parts
        d['svms'] = self._svms
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'],d['part_shape'],d['settings'])
        obj._train_info = d['training_info']
        obj._partition = d['partition']
        obj._parts = d['parts']
        obj._svms = d['svms']
        return obj





#To future speed up, we want each processors to take care of svms.
def task_spread((patches_label, partition, i, round_each_proc)):
    svm_objects = []
    for j in range(round_each_proc * i, round_each_proc * (i + 1)):
        svm_objects.append(multi_sgd_train((shared_data, patches_label, partition[j])))
    return svm_objects



def multi_sgd_train((data, patches_label, partition)):
    data = data.reshape((data.shape[0], -1))
    assert data.shape[0]==patches_label.shape[0]
    from sklearn import linear_model
    from sklearn import svm
    clf = linear_model.SGDClassifier(n_iter=2)
    #clf = svm.LinearSVC()
    partitionLabel = [partition[i] for i in patches_label]
    clf.fit(data, partitionLabel)
    print(partition)
    print(np.mean(clf.predict(data) == partitionLabel))
    return clf

def calcScore((svm, data)):
    result = np.array(svm.decision_function(data),dtype=np.float16)
    result = result.reshape((result.shape[0], 1))
    return result

def task_spread_svms((i, round_each_proc, part_shape, sample_shape, class1, class2)):
    svm_objects = []
    for j in range(round_each_proc * i, round_each_proc * (i + 1)):
        #everytime we pick only one location,
        print i
        patches, patches_label = get_color_patches_location(shared_X, shared_Y, class1[j - i * round_each_proc], class2[j - i * round_each_proc],
                                                            locations_per_try=1, part_shape=part_shape,
                                                            sample_shape=sample_shape,fr = 1, randomseed=j,
                                                            threshold=0, max_samples=300000)
        #combined = zip(patches,patches_label)
        #import random
        #random.shuffle(combined)
        #patches[:], patches_label[:] = zip(*combined)

        from sklearn import linear_model
        import sklearn
        #from sklearn.svm import LinearSVC
        from sklearn import svm
        #print("max values .....")
        #print(np.max(patches))
        #print("means.....")
        #print(np.mean(patches, axis = 0))
        clf = linear_model.SGDClassifier(loss = "hinge",penalty = 'l2', n_iter=200, shuffle=True,verbose = False,
                                         learning_rate = "constant", eta0 = 0.01, random_state = 0)
        #clf = LinearSVC(C=1.0)
        #X_scaled = sklearn.preprocessing.scale(patches)
        clf.fit(patches, patches_label)
        print(np.mean(clf.predict(patches) == patches_label))
        svm_objects.append(clf)
    return svm_objects

def split_1vs2(n):
    import itertools
    class1 = []
    class2 = []
    for i in range(n):
        restSet = [j for j in range(n) if j!=i]
        subset = list(itertools.combinations(restSet, 2))
        class2+=subset
        class1+=([i] for j in range(len(subset)))
    return class1, class2





def get_color_patches_location(X, Y, class1, class2, locations_per_try, part_shape, sample_shape, fr, randomseed, threshold, max_samples):
    assert X.ndim == 4
    channel = X.shape[-1]
    patches = []
    patches_label = []
    rs = np.random.RandomState(randomseed)

    #th = self._settings['threshold']
    th = threshold
    w, h = [X.shape[1 + j]-part_shape[j] - sample_shape[j] +2 for j in range(2)]
    indices = list(itr.product(range(w-1), range(h-1)))
    rs.shuffle(indices)
    i_iter = itr.cycle(iter(indices))
    for trie in range(locations_per_try):
        x, y = next(i_iter)
        print(x,y)
        for i in range(X.shape[0]):
            Xi = X[i]
            # How many patches could we extract?
            if(Y[i] not in class1 and Y[i] not in class2):
                continue
            for x_i in range(sample_shape[0]):
                for y_i in range(sample_shape[1]):
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
                            patches_label.append(-1)
                        else:
                            patches_label.append(1)
                        if len(patches) >= max_samples:
                            patches = np.asarray(patches)
                            patches = patches.reshape((patches.shape[0], -1))
                            return np.asarray(patches),np.asarray(patches_label).astype(np.uint8)

    ag.info('ENDING: {} patches'.format(len(patches)))
    patches = np.asarray(patches)
    patches = patches.reshape((patches.shape[0], -1))
    return np.asarray(patches),np.asarray(patches_label).astype(np.uint8)


