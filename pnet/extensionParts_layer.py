from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet

@Layer.register('rotatable_extensionParts_layer')
class ExtensionPartsLayer(Layer):
    def __init__(self, num_parts, num_components, part_shape, settings={}):
        self._num_parts = num_parts * num_components
        self._num_lower_parts = num_parts
        self._num_components = num_components
        self._part_shape = part_shape
        self._settings = settings

        self._train_info = {}
        self._parts = None
        self._weights = None

    def extract(self,X):
        assert self._parts is not None, "Must be trained before calling extract"
        assert X.ndim == 4, "Input X dimension is not correct"
        extractedFeature = np.empty((X.shape[0],X.shape[1] - self._part_shape[0],X.shape[2] - self._part_shape[1]))
        for i in range(X.shape[0]):
            patch = X[i]
            for m in range(patch.shape[0] - self._part_shape[0] + 1):
                for n in range(patch.shape[1] - self._part_shape[1] + 1):
                    region = patch[m:m+self._part_shape[0],n:n+self._part_shape[1]]                    
                    centerPart = region[self._part_shape[0]/2 + 1, self._part_shape[1]/2 + 1]
                    extractedFeature[i,m,n] = self._parts[centerPart][0].extract(partsPool(region))[0] + centerPart * self._num_components
        return extractedFeature,self._num_parts
        
    
    @property
    def trained(self):
        return self._parts is not None 

    def train(self, X, Y=None, OriginalX = None):
        assert Y is None
        ag.info('Extracting patches')
        patches, patches_original = self._get_patches(X,OriginalX)
        processedPatches = self._process_patches(patches)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self._train_from_processed_patches(processedPatches)


    def _process_patches(self,patches):
        partsRegion = [[] for x in range(self._num_lower_parts)]
        for eachPatch in patches:
            #Find out the part that coded in the center
            codePart = np.argmax(eachPatch[self._part_shape[0]/2 - 1, self._part_shape[1]/2 - 1])
            print(codePart, codePart.shape)
            #Put this patch to the belong the list
            partsRegion[codePart].append(partsPool(eachPatch,self._num_lower_parts))
        return partsRegion
    
    def partsPool(originalPartsRegion, numParts):
        partsGrid = np.zeros((1,1,numParts))
        for i in range(originalPartsRegion.shape[0]):
            for j in range(originalPartsRegion.shape[1]):
                if(originalPartsRegion[i,j]!=-1):
                    partsGrid[0,0,originalPartsRegion[i,j]] = 1
        return partsGrid
                    
    def _train_from_processed_patches(self, processedPatches):
        allPartsLayer = [[pnet.PartsLayer(self._num_components,(1,1),settings = dict(outer_frame = 0, threshold = 5, sample_per_image = 1, max_samples = 10000, min_prob = 0.005))] for i in range(self._num_lower_parts)]
        for i in range(self._num_lower_parts):
            if(not processedPatches):
                continue
            allPartsLayer[i][0].train_from_samples(np.array(processedPatches[i],None))
        self._parts = allPartsLayer


    def _get_patches(self, X, OriginalX):
        assert X.ndim == 4
        assert OriginalX.ndim == 3
        samples_per_image = self._settings.get('samples_per_image', 20) 
        fr = self._settings['outer_frame']
        patches = []
        patches_original = []
        rs = np.random.RandomState(self._settings.get('patch_extraction_seed', 0))

        th = self._settings['threshold']

        for i in range(X.shape[0]):
            Xi = X[i]
            OriginalXi = OriginalX[i]
            # How many patches could we extract?
            w, h = [Xi.shape[i]-self._part_shape[i]+1 for i in range(2)]

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
                        tot = patch[fr:-fr,fr:-fr].sum()

                    if th <= tot: 
                        patches.append(patch)
                        vispatch = OriginalXi[selection]
                        span = vispatch.min(),vispatch.max()
                        if span[1]-span[0] > 0:
                            vispatch = (vispatch - span[0])/(span[1] - span[0])
                        patches_original.append(vispatch)
                        if len(patches) >= self._settings.get('max_samples', np.inf):
                            return np.asarray(patches),np.asarray(patches_original)
                        break

                    if tries == N-1:
                        ag.info('WARNING: {} tries'.format(N))

        return np.asarray(patches),np.asarray(patches_original)

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
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['parts'] = self._parts
        d['weights'] = self._weights
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._parts = d['parts']
        obj._weights = d['weights']
        return obj
