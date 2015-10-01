__author__ = 'jiajunshen'
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
from multiprocessing import Pool, Value, Array


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
        patches, patches_label, patches_original = self._get_color_patches(X,Y,OriginalX)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        ag.info('Training patches labels', patches_label.shape)
        return self.train_from_samples(patches,patches_label, patches_original)

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

        for x in range(dim[0]):
            for y in range(dim[1]):
                Xr = X[:,x:x+self._part_shape[0],y:y+self._part_shape[1],:]
                Xr_flat = Xr.reshape((X.shape[0], -1)).astype(np.float16)
                args = ((self._svms[i],Xr_flat) for i in range(self._num_parts))
                scores = p.map(calcScore, args)
                scores = np.hstack(scores)
                feature_map[:,x, y,:] = scores.astype(np.float16)

        print(feature_map.shape)

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
        np.random.seed(self._settings.get('random_seed', 0))
        for i in range(self._num_parts):
            sampledGroup = np.zeros(10).astype(np.uint8)
            sampledGroup[np.random.choice(range(10),size = sampleNumber,replace=False)] = 1
            map = dict(enumerate(sampledGroup))
            self._partition.append(map)
        return

    def train_from_samples(self, patches, patches_label, patches_original):
        ag.info("training from samples starts")
        self.random_partition()
        import time
        p = Pool(4)
        #patches_list = [patches for i in range(4)]

        args = ((patches,)+(patches_label,)+(self._partition[i],) for i in range(self._num_parts))
        start_time = time.time()
        svm_objects = p.map(multi_sgd_train, args)
        print("--- %s seconds ---" % (time.time() - start_time))
        print len(svm_objects)
        self._svms = svm_objects
        self._parts = []
        for i in range(self._num_parts):
            svm_coef = svm_objects[i].coef_
            svm_intercept = svm_objects[i].intercept_
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




"""
#To future speed up, we want each processors to take care of svms.
def task_spread((data, patches_label, partition, i)):
    svm_objects = []
    for j in range(8 * i, 8* (i + 1)):
        print j
        svm_objects.append(multi_sgd_train((data, patches_label, partition[j])))
    return svm_objects
"""


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


