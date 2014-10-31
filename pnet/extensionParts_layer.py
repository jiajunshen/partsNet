from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
from pnet.cyfuncs import index_map_pooling

@Layer.register('extensionParts_layer')
class ExtensionPartsLayer(Layer):
    def __init__(self, num_parts, num_components, part_shape, lowerLayerShape, settings={}):
        self._num_parts = num_parts * num_components
        self._num_lower_parts = num_parts
        self._num_components = num_components
        self._part_shape = part_shape
        self._lowerLayerShape = lowerLayerShape
        self._settings = settings
        self._classificationLayers = None
        self._train_info = {}

    def extract(self,X):
        assert self._classificationLayers is not None, "Must be trained before calling extract"
        X, num_parts = X
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
        
        for i in range(X.shape[0]):
            patch = X[i]
            for m in range(patch.shape[0] + self._lowerLayerShape[0]- self._part_shape[0]):
                for n in range(patch.shape[1] + self._lowerLayerShape[1] - self._part_shape[1]):
                    if(secondLevelCurxCenter[i,m,n]!=-1):
                        firstLevelPartIndex = secondLevelCurxCenter[i,m,n]
                        firstLevelPartIndex = int(firstLevelPartIndex)
                        extractedFeaturePart = self.codeParts(np.array(secondLevelCurx[i,m,n][np.newaxis,:], dtype = np.uint8), self._classificationLayers[firstLevelPartIndex])[0]
                        extractedFeature[i,m,n] = int(self._num_components * firstLevelPartIndex + extractedFeaturePart)
                    else:
                        extractedFeature[i,m,n] = -1
        extractedFeature = np.array(extractedFeature[:,:,:,np.newaxis],dtype = np.int64)
        return (extractedFeature,self._num_parts)
        
    def codeParts(self,ims, allLayers):
        curX = ims
        for layer in allLayers:
            curX = layer.extract(curX)
        return curX

    @property
    def trained(self):
        return self._classificationLayers is not None 

    def train(self, X, Y=None, OriginalX = None):
        
        X_data, num_lower_parts = X
        

        assert Y is None
        
        assert X_data.ndim == 4

        assert num_lower_parts == self._num_lower_parts
        
        X_data = X_data.reshape((X_data.shape[0],X_data.shape[1],X_data.shape[2]))
        #Extract Patches
        partsRegion = self._extract_patches(X_data)
        
        #Train Patches 
        ag.info('Done extracting patches')
        #ag.info('Training patches', patches.shape)
        return self._train_patches(partsRegion)


    def _extract_patches(self,X):
        
        ag.info('Extract Patches: Seperate coded regions into groups')
        trainingDataNum = X.shape[0]
        totalRange = X.shape[1]
        frame = (self._part_shape[0] - self._lowerLayerShape[0]) / 2
        frame = int(frame)
        partsRegion = [[] for x in range(self._num_lower_parts)]

        
        for i in range(trainingDataNum):
            for m in range(totalRange)[frame:totalRange - frame]:
                for n in range(totalRange)[frame:totalRange - frame]:
                    if(X[i,m,n]!=-1):
                        partsGrid = self.partsPool(X[i,m-frame:m+frame+1,n-frame:n+frame+1], self._num_lower_parts)
                        partsRegion[X[i,m,n]].append(partsGrid)
        return partsRegion

    def _train_patches(self,partsRegion):
        allPartsLayer = [[pnet.PartsLayer(self._num_components,(1,1),
                    settings=dict(outer_frame = 0,
                    em_seed = self._settings.get('em_seed',0),
                    threshold = 5,
                    sample_per_image = 1,
                    max_samples=10000,
                    min_prob = 0.005,
                    #min_llh = -40
                    ))]
                    for i in range(self._num_lower_parts)]
        for i in range(self._num_lower_parts):
            if(not partsRegion[i]):
                continue
            allPartsLayer[i][0].train_from_samples(np.array(partsRegion[i]),None)
        self._classificationLayers = allPartsLayer
    
    def partsPool(self,originalPartsRegion, numParts):
        partsGrid = np.zeros((1,1,numParts))
        for i in range(originalPartsRegion.shape[0]):
            for j in range(originalPartsRegion.shape[1]):
                if(originalPartsRegion[i,j]!=-1):
                    partsGrid[0,0,originalPartsRegion[i,j]] = 1
        return partsGrid


    def infoplot(self, vz):
        from pylab import cm
        D = self._parts.shape[-1]
        N = self._num_parts
        # Plot all the parts
        grid = pnet.plot.ImageGrid(N, D, self._part_shape)

        print('SHAPE', self._parts.shape)

        cdict1 = {'red':  ((0.0, 0.0, 0.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),

                 'green': ((0.0, 0.4, 0.4),
                           (0.5, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),

                 'blue':  ((0.0, 1.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.4, 0.4))
                }

        from matplotlib.colors import LinearSegmentedColormap
        C = LinearSegmentedColormap('BlueRed1', cdict1)

        for i in range(N):
            for j in range(D):
                grid.set_image(self._parts[i,...,j], i, j, cmap=C)#cm.BrBG)

        grid.save(vz.generate_filename(), scale=5)

        #vz.log('weights:', self._weights)
        #vz.log('entropy', self._train_info['entropy'])

        import pylab as plt
        plt.figure(figsize=(6, 3))
        plt.plot(self._weights, label='weight')
        plt.savefig(vz.generate_filename(ext='svg'))

        vz.log('weights span', self._weights.min(), self._weights.max()) 

        import pylab as plt
        plt.figure(figsize=(6, 3))
        plt.plot(self._train_info['entropy'])
        plt.savefig(vz.generate_filename(ext='svg'))

        vz.log('median entropy', np.median(self._train_info['entropy']))

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['num_lower_parts'] = self._num_lower_parts
        d['num_components'] = self._num_components
        d['part_shape'] = self._part_shape
        d['lowerLayerShape'] = self._lowerLayerShape
        d['settings'] = self._settings
        d['classificationLayers'] = self._classificationLayers
        d['train_info'] = self._train_info
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_lower_parts'], d['num_components'],d['part_shape'], d['lowerLayerShape'], settings=d['settings'])
        obj._classificationLayers = d['classificationLayers']
        obj._train_info = d['train_info']
        return obj
