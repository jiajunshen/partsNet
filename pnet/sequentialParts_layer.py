from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet

@Layer.register('sequenctialParts_layer')
class SequentialPartsLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._settings = settings
        self._train_info = {}
        self._parts = None
        self._weights = None

    def extract(self,X):
        assert self._parts is not None, "Must be trained before calling extract"
        th = self._settings['threshold']
        part_logits = np.rollaxis(logit(self._parts).astype(np.float64),0,4)
        constant_terms = np.apply_over_axes(np.sum, np.log(1-self._parts).astype(np.float64), [1, 2, 3]).ravel()

        from pnet.cyfuncs import code_index_map_multi

        feature_map = code_index_map_multi(X, part_logits, constant_terms, th,
                                           outer_frame=self._settings['outer_frame'], 
                                           min_llh=self._settings.get('min_llh', -np.inf),
                                           n_coded=self._settings.get('n_coded', 1))


        return (feature_map, self._num_parts)
    @property
    def trained(self):
        return self._parts is not None 

    def train(self, X, Y=None, OriginalX = None):
        assert Y is None
        ag.info('Extracting patches')
        patches, patches_original = self._get_patches(X,OriginalX)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.sequential_train_from_samples(patches)


    def sequential_train_from_samples(self,patches):
        allParts = np.ones((self._num_parts,)+patches.shape[1:])
        currentPatches = patches
        print(patches.shape)
        for i in range(self._num_parts):
            numParts = self._num_parts - i
            currentParts = self.train_from_samples(currentPatches,numParts)
            allParts[i] = currentParts[0]
            codeIndex = self.codePatches(currentPatches,currentParts)
            currentPatches = currentPatches[codeIndex!=0]
            print(currentPatches.shape)
            if(currentPatches.shape[0] <= 0):
                break;
        self._parts = allParts


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
        

    def train_from_samples(self, patches,num_parts):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        from pnet.bernoullimm import BernoulliMM
        min_prob = self._settings.get('min_prob', 0.01)
        flatpatches = patches.reshape((patches.shape[0], -1))
        parts = np.ones((num_parts,) + patches.shape[1:])
        if 0:
            mm = BernoulliMM(n_components=num_parts, n_iter=20, tol=1e-15,n_init=2, random_state=0, min_prob=min_prob, verbose=False)
            print(mm.fit(flatpatches))
            print('AIC', mm.aic(flatpatches))
            print('BIC', mm.bic(flatpatches))
            #import pdb; pdb.set_trace()
            parts = mm.means_.reshape((num_parts,)+patches.shape[1:])
            #self._weights = mm.weights_
        else:
            
            from pnet.bernoulli import em
            ret = em(flatpatches, num_parts,20,numpy_rng=self._settings.get('em_seed',0),verbose=True)
            parts = ret[1].reshape((num_parts,) + patches.shape[1:])
            self._weights = np.arange(self._num_parts)

            #self._weights = mm.weights
            
        # Calculate entropy of parts
        Hall = (parts * np.log(parts) + (1 - parts) * np.log(1 - parts))
        H = -np.apply_over_axes(np.mean, Hall, [1, 2, 3])[:,0,0,0]

        # Sort by entropy
        II = np.argsort(H)

        parts[:] = parts[II]
        #self._train_info['entropy'] = H[II]
        return parts

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
