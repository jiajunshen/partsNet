from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
from pnet.cyfuncs import index_map_pooling
import pnet

@Layer.register('rotatable_extensionParts_layer')
class RotExtensionPartsLayer(Layer):
    def __init__(self, num_parts, num_components, part_shape, rotation = None, lowerLayerShape= None, shifting = None, settings={}):
        self._rotation = rotation
        self._shifting = shifting
        self._num_parts = num_parts  * self._rotation * num_components
        self._num_lower_parts = num_parts * self._rotation
        self._num_true_parts = self._num_parts // num_components
        self._num_components = num_components
        self._part_shape = part_shape
        self._lowerLayerShape = lowerLayerShape
        self._settings = settings
        self._models = None
        self._train_info = {}
        self._partsDistance = None

    def extract(self,X):
        assert self._models is not None, "Must be trained before calling extract"
        if(len(X) == 2):
            X, num_parts = X
        else:
            X, num_parts, orientation = X
        
        assert X.ndim == 4, "Input X dimension is not correct"
        assert num_parts == self._num_lower_parts

        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        secondLevelCurx = np.zeros((X.shape[0], X.shape[1] + self._lowerLayerShape[0] - self._part_shape[0], X.shape[2] + self._lowerLayerShape[1] - self._part_shape[1],1,1,self._num_lower_parts))
        secondLevelCurxCenter = np.zeros((X.shape[0], X.shape[1] + self._lowerLayerShape[0] - self._part_shape[0], X.shape[2] + self._lowerLayerShape[1] - self._part_shape[1]))

        extractedFeature = np.empty((X.shape[0],X.shape[1] + self._lowerLayerShape[0] - self._part_shape[0],X.shape[2] + self._lowerLayerShape[1] - self._part_shape[1]))
        totalRange = X.shape[1]
        frame = (self._part_shape[0] - self._lowerLayerShape[0]) / 2
        frame = int(frame)

        for m in range(totalRange)[frame:totalRange-frame]:
            for n in range(totalRange)[frame:totalRange-frame]:
                secondLevelCurx[:,m-frame,n-frame] = index_map_pooling(X[:,m-frame:m+frame+1,n-frame:n+frame+1],self._num_lower_parts, (2 * frame + 1, 2 * frame + 1), (2 * frame + 1, 2 * frame + 1))
                secondLevelCurxCenter[:,m-frame,n-frame] = X[:,m,n]

        for m in range(X[0].shape[0] + self._lowerLayerShape[0]- self._part_shape[0]):
            for n in range(X[0].shape[1] + self._lowerLayerShape[1] - self._part_shape[1]):
                codedIndex = np.where(secondLevelCurxCenter[:,m,n]!=-1)[0]
                notCodedIndex = np.where(secondLevelCurxCenter[:,m,n] == 0)[0]
                print(codedIndex.shape)
                
                firstLevelPartIndex = secondLevelCurxCenter[codedIndex,m,n]
                firstLevelPartIndex = np.array(firstLevelPartIndex,dtype = np.int)
                firstLevelPartIndex = (firstLevelPartIndex - firstLevelPartIndex % self._rotation)//self._rotation
                theta = self._models[firstLevelPartIndex]
                XX = secondLevelCurx[codedIndex, m, n][:,np.newaxis]
                print(XX.shape, theta.shape)
                llh = XX * np.log(theta) + (1 - XX) * np.log(1 - theta)
                bb = np.apply_over_axes(np.sum, llh, [-3, -2, -1])[...,0,0,0]
                #print(bb.shape)
                maxIndex = np.argmax(bb,axis = -1)
                print(maxIndex.shape)
                extensionPart = maxIndex // self._rotation
                rotatedAngle = maxIndex % self._rotation
                extractedFeature[codedIndex,m,n] = np.array(firstLevelPartIndex * self._num_components * self._rotation + extensionPart * self._rotation + rotatedAngle, dtype = np.int)
                extractedFeature[notCodedIndex,m,n] = -1
        extractedFeature = np.array(extractedFeature[:,:,:,np.newaxis],dtype = np.int64)
        return (extractedFeature,self._num_parts, self._rotation)
        

    @property
    def trained(self):
        return self._models is not None 

    def train(self, X_n, Y=None, OriginalX = None):
        assert Y is None
        X = X_n[0]
        num_parts = X_n[1]
        num_orientations = X_n[2]
        num_true_parts = num_parts // num_orientations
        print(num_parts,num_orientations,num_true_parts, self._num_parts)
        assert num_parts  * self. _num_components == self._num_parts
        assert num_orientations == self._rotation
        
        assert X.ndim == 4
        X = X.reshape((X.shape[0],X.shape[1],X.shape[2]))

        ag.info('Extracting patches')


        partsRegion, shape = self._extract_patches(X)
        ag.info('Done extracting patches')
        return self._train_patches(partsRegion,shape)

    def _extract_patches(self,X):
        
        ag.info('Extract Patches: Seperate coded regions into groups')
        trainingDataNum = X.shape[0]
        totalRange = X.shape[1]
        frame = (self._part_shape[0] - self._lowerLayerShape[0]) / 2
        frame = int(frame)
        partsRegion = [[] for x in range(self._num_lower_parts//self._rotation)]
        
        
        for i in range(trainingDataNum):
            for m in range(totalRange)[frame:totalRange - frame]:
                for n in range(totalRange)[frame:totalRange - frame]:
                    if(X[i,m,n]!=-1):
                        partsGrid = X[i,m-frame:m+frame+1,n-frame:n+frame+1]
                        detectedIndex = X[i,m,n]
                        uprightIndex = (detectedIndex - detectedIndex % self._rotation)//self._rotation
                        partsRegion[uprightIndex].append(partsGrid)
        finalResultRegion = []
        for i in range(self._num_lower_parts//self._rotation):
            blocks = []
            print(np.asarray(partsRegion[i]).shape)
            for ori in range(self._rotation):
                angle = ori / self._rotation * 360
                from pnet.cyfuncs import rotate_index_map_pooling
                yy1 = rotate_index_map_pooling(np.asarray(partsRegion[i]),angle, 0, self._rotation, self._num_lower_parts, (self._part_shape[0] - self._lowerLayerShape[0] + 1,self._part_shape[1] - self._lowerLayerShape[1] + 1))
                #yy.shape = (numOfPatches, 1, 1, rotation, trueParts)
                yy = yy1.reshape(yy1.shape[:3] + (self._rotation, self._num_lower_parts // self._rotation))
                blocks.append(yy)
            #blocks.shape = (numofPatches, numOfRotations, 1, 1, rotation, trueParts)
            blocks = np.asarray(blocks).transpose((1,0,2,3,4,5))
            shape = blocks.shape[2:4] + (np.prod(blocks.shape[4:]),)
            ## Flatten
            blocks = blocks.reshape(blocks.shape[:2] + (-1,))
            finalResultRegion.append(blocks)
        return (finalResultRegion, shape)


        

    def _train_patches(self,partsRegion,shape):
        ORI = self._rotation
        def cycles(X):
            return np.asarray([np.concatenate([X[i:],X[:i]]) for i in xrange(len(X))])
        RR = np.arange(ORI)
        POL = 1
        PP = np.arange(POL)
        II = [list(itr.product(PPi, RRi)) for PPi in cycles(PP) for RRi in cycles(RR)]
        lookup = dict(zip(itr.product(PP,RR),itr.count()))
        num_angle = self._rotation
        d = np.prod(shape)
        permutation = np.empty((num_angle, num_angle * d), dtype = np.int_)
        for a in range(num_angle):
            if a == 0:
                permutation[a] = np.arange(num_angle * d)
            else:
                permutation[a] = np.roll(permutation[a-1], -d)
        from pnet.bernoulli import em
        mm_models = []
        for i in range(self._num_lower_parts//self._rotation):
            block = partsRegion[i]
            ret = em(block.reshape((block.shape[0],-1)),self._num_components, 10, permutation = permutation, numpy_rng = 0, verbose = True)
            mu = ret[1].reshape((self._num_components * self._rotation, ) + shape)
            mm_models.append(mu)
        self._models = np.asarray(mm_models)




    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['rotation'] = self._rotation
        d['shifting'] = self._shifting
        d['num_lower_parts'] = self._num_lower_parts
        d['num_components'] = self._num_components
        d['part_shape'] = self._part_shape
        d['lowerLayerShape'] = self._lowerLayerShape
        d['settings'] = self._settings
        d['models'] = self._models
        d['train_info'] = self._train_info
        d['partsDistance'] = self._partsDistance
        return d

    @classmethod
    def load_from_dict(cls, d):
        numParts = d['num_lower_parts'] // d['rotation']
        obj = cls(numParts, d['num_components'],d['part_shape'], d['rotation'],d['lowerLayerShape'], settings=d['settings'])
        obj._models = d['models']
        obj._train_info = d['train_info']
        obj._partsDistance = d['partsDistance']
        return obj
