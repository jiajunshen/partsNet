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
        self._classificationLayers = None
        self._train_info = {}
        self._partsDistance = None

    def extract(self,X):
        assert self._classificationLayers is not None, "Must be trained before calling extract"
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
                for i in range(X.shape[0]):
                    secondLevelCurx[i,m-frame,n-frame] = self._process_testing_patches(secondLevelCurx[i,m-frame,n-frame],secondLevelCurxCenter[i,m-frame,n-frame])

        for i in range(X.shape[0]):
            patch = X[i]
            for m in range(patch.shape[0] + self._lowerLayerShape[0]- self._part_shape[0]):
                for n in range(patch.shape[1] + self._lowerLayerShape[1] - self._part_shape[1]):
                    if(secondLevelCurxCenter[i,m,n]!=-1):
                        firstLevelPartIndex = secondLevelCurxCenter[i,m,n]
                        firstLevelPartIndex = int(firstLevelPartIndex)
                        firstLevelRotation = firstLevelPartIndex % self._rotation
                        extractedFeaturePart = self.codeParts(np.array(secondLevelCurx[i,m,n][np.newaxis,:], dtype = np.uint8), self._classificationLayers[(firstLevelPartIndex - firstLevelRotation) // self._rotation])[0]
                        extractedFeature[i,m,n] = int((firstLevelPartIndex - firstLevelRotation) // self._rotation * self._num_components * self._rotation + extractedFeaturePart * self._rotation + firstLevelRotation)
                        #extractedFeature[i,m,n] = int(self._num_components * firstLevelPartIndex + extractedFeaturePart)
                    else:
                        extractedFeature[i,m,n] = -1
        extractedFeature = np.array(extractedFeature[:,:,:,np.newaxis],dtype = np.int64)
        return (extractedFeature,self._num_parts, self._rotation)
        
    def codeParts(self,ims, allLayers):
        curX = ims
        for layer in allLayers:
            curX = layer.extract(curX)
        return curX

    @property
    def trained(self):
        return self._classificationLayers is not None 

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


        partsRegion = self._extract_patches(X)
        patches = self._process_patches(partsRegion)
        ag.info('Done extracting patches')
        #ag.info('Training patches', patches.shape)
        return self._train_patches(patches)

    def partsPool(self,originalPartsRegion, numParts):
        partsGrid = np.zeros((1,1,numParts))
        for i in range(originalPartsRegion.shape[0]):
            for j in range(originalPartsRegion.shape[1]):
                if(originalPartsRegion[i,j]!=-1):
                    partsGrid[0,0,originalPartsRegion[i,j]] = 1
        return partsGrid
    
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

    def _process_patches(self, partsRegion):
        ## This is to rotate the image patches back
        processedPartsRegion = [[] for x in range(self._num_lower_parts // self._rotation)]
        assert len(partsRegion) == self._num_lower_parts
        for i in range(self._num_lower_parts):
            if i % self._rotation == 0:
                processedPartsRegion[i//self._rotation] = partsRegion[i]
            else:
                rotation = i % self._rotation
                for partsGrid in partsRegion[i]:
                    newPartsGrid = np.zeros(partsGrid.shape)
                    for codeIndex in range(self._num_lower_parts):
                        if partsGrid[0, 0,codeIndex] == 1:
                            startIndex = codeIndex - codeIndex % self._rotation
                            lowerPartsRotation = codeIndex % self._rotation
                            newPartsGrid[0, 0, int(startIndex + (lowerPartsRotation + rotation) % self._rotation)] = 1
                    processedPartsRegion[(i - rotation)//self._rotation].append(partsGrid)
        return processedPartsRegion

    def _process_testing_patches(self, partsGrid, rotation):
        if rotation % self._rotation == 0:
            return partsGrid
        newPartsGrid = np.zeros(partsGrid.shape)
        for codeIndex in range(self._num_lower_parts):
            if partsGrid[0,0,codeIndex] == 1:
                startIndex = codeIndex - codeIndex % self._rotation
                lowerPartsRotation = codeIndex % self._rotation
                newPartsGrid[0,0,int(startIndex + (lowerPartsRotation + rotation) % self._rotation)] = 1
        return newPartsGrid


    def codePatches(self,patches,currentParts):
        flatpatches = patches.reshape((patches.shape[0],-1))
        print(flatpatches.shape)
        part_logits = np.rollaxis(logit(currentParts).astype(np.float64),0,4)
        part_logits = part_logits.reshape(part_logits.shape[0] * part_logits.shape[1] * part_logits.shape[2], -1)
        print(part_logits.shape)
        constant_terms = np.apply_over_axes(np.sum, np.log(1-currentParts).astype(np.float64),[1,2,3]).ravel()
        print(constant_terms.shape)
        codeParts = np.dot(flatpatches,part_logits)
        codeParts = codeParts + constant_terms
        print(codeParts.shape)
        return np.argmax(codeParts, axis = 1)
        

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
                    for i in range(self._num_lower_parts//self._rotation)]
        for i in range(self._num_lower_parts//self._rotation):
            if(not partsRegion[i]):
                continue
            allPartsLayer[i][0].train_from_samples(np.array(partsRegion[i]),None)
        self._classificationLayers = allPartsLayer



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
        d['rotation'] = self._rotation
        d['shifting'] = self._shifting
        d['num_lower_parts'] = self._num_lower_parts
        d['num_components'] = self._num_components
        d['part_shape'] = self._part_shape
        d['lowerLayerShape'] = self._lowerLayerShape
        d['settings'] = self._settings
        d['classificationLayers'] = self._classificationLayers
        d['train_info'] = self._train_info
        d['partsDistance'] = self._partsDistance
        return d

    @classmethod
    def load_from_dict(cls, d):
        numParts = d['num_lower_parts'] // d['rotation']
        obj = cls(numParts, d['num_components'],d['part_shape'], d['rotation'],d['lowerLayerShape'], settings=d['settings'])
        obj._classificationLayers = d['classificationLayers']
        obj._train_info = d['train_info']
        obj._partsDistance = d['partsDistance']
        return obj
