from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet

@Layer.register('moduloShiftingParts_layer')
class ModuloShiftingPartsLayer(Layer):
    def __init__(self, num_parts, part_shape,shifting_shape, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._shifting_shape = shifting_shape
        self._settings = settings
        self._train_info = {}
        self._parts = None
        self._weights = None
        self._visparts = None
        self._bkg_probability = None
        self._sample_shape = (part_shape[0] + shifting_shape[0] - 1, part_shape[1] + shifting_shape[1] - 1)

    def _extract_many_edges(self,bedges_settings, settings, images, must_preserve_size=False):
        """Extract edges of many images (must be the same size)"""
        sett = bedges_settings.copy()
        sett['radius'] = 0
        sett['preserve_size'] = False or must_preserve_size

        edge_type = settings.get('edge_type', 'yali')
        if edge_type == 'yali':
            X = ag.features.bedges(images, **sett)
            return X
        else:
            raise RuntimeError("No such edge type")


    def _extract_batch(self,im, settings, num_parts, num_permutation, part_shape, parts):
        assert parts is not None, "must be trained before calling extract"


        x = im
        print(parts.shape)
        print(x.shape) 
        th = settings['threshold']
        n_coded = settings.get('n_coded', 1)
        
        support_mask = np.ones(part_shape, dtype=np.bool)

        from pnet.cyfuncs import code_index_map_general
        #print("=================")
        #print(x.shape)
        #print("=================")

        feature_map = code_index_map_general(x, 
                                             parts, 
                                             support_mask.astype(np.uint8),
                                             th,
                                             outer_frame=settings['outer_frame'], 
                                             n_coded=n_coded,
                                             standardize=settings.get('standardize', 0),
                                             min_percentile=settings.get('min_percentile', 0.0))

        # shifting spreading?
        print(feature_map.shape) 
        return feature_map


    
#    def extract(self,X):
#        #assert self._parts is not None, "Must be trained before calling extract"
#        th = self._settings['threshold']
#        b = int(X.shape[0] // 100)
#        sett = (self._settings, self._num_parts, self._shifting_shape[0] * self._shifting_shape[1], self._part_shape,self._parts)
#        if 1: #b == 0:
#            feat = self._extract_batch(X, *sett)
#        else:
#            im_batches = np.array_split(X,b)
#
#            args = ((im_b,) + sett for im_b in im_batches)
#
#            feat = np.concatenate([batch for batch in pnet.parallel.starmap_unordered(_extract_batch, args)])
#        return (feat, self._num_parts, self._shifting_shape)
#"""
    def extract(self,X):
        print("+++++++++++++++++++++++++++++++++++++++++")
        assert self._parts is not None, "Must be trained before calling extract"
        assert self._bkg_probability is not None, "bkg probability is None"
        from pnet.latentShiftEM import Extract
        dataShape = X.shape
        print(X.shape)
        block_size = 400
        finalResult = np.empty((dataShape[0],dataShape[1]-self._sample_shape[0] +1 , dataShape[2] - self._sample_shape[1] + 1,1))
        num_block = int(np.ceil(dataShape[0]/block_size))
        print(self._sample_shape,self._part_shape)
        for p in range(num_block):
            print("inside",p)
            print(X.shape)
            #print(X[i * block_size:(i + 1) * block_size,slice(i,i+self._sample_shape[0]),slice(j,j+self._sample_shape[1]),:].shape)
            print(np.array([X[p * block_size:(p + 1) * block_size,slice(i,i+self._sample_shape[0]),slice(j,j+self._sample_shape[1]),:] for i in range(dataShape[1] - self._sample_shape[0] + 1) for j in range(dataShape[2] - self._sample_shape[1] + 1)]).shape)
            x_patches = np.swapaxes(np.array([X[p * block_size:(p + 1) * block_size,slice(i,i+self._sample_shape[0]),slice(j,j+self._sample_shape[1]),:] for i in range(dataShape[1] - self._sample_shape[0] + 1) for j in range(dataShape[2] - self._sample_shape[1] + 1)]),0,1)
            x_patches_shape = x_patches.shape
            print(x_patches_shape)
            x_patches = x_patches.reshape((x_patches.shape[0] * x_patches.shape[1], -1))
            result = Extract(data = x_patches, num_mixture_component = self._num_parts,parts_shape = (self._part_shape[0],self._part_shape[1],8),region_shape = (self._sample_shape[1],self._sample_shape[1],8),shifting_shape = self._shifting_shape, mu = self._parts.reshape(self._num_parts,-1),bkg_probability = self._bkg_probability)[:,0]
            result = result.reshape((x_patches_shape[0],x_patches_shape[1]))
            print(result.shape)
            result = result.reshape((x_patches_shape[0], (dataShape[1] - self._sample_shape[0] + 1) , (dataShape[2] - self._sample_shape[1] + 1), 1))
            print(result.shape)
            finalResult[p*block_size:(p+1)*block_size] = result
        finalResult = np.array(finalResult, dtype = np.int64)
        return finalResult, self._num_parts


    @property
    def trained(self):
        return self._parts is not None 

    def train(self, X, Y=None, OriginalX = None):
        assert Y is None
        ag.info('Extracting patches')
        patches, patches_original = self._get_patches(X,OriginalX)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches, patches_original)

        

    def train_from_samples(self, patches, original_patches):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        from pnet.bernoullimm import BernoulliMM
        print(patches.shape)
        min_prob = self._settings.get('min_prob', 0.01)
        #num_permutation = self._shifting_shape[0] * self._shifting_shape[1]
        #parts = np.ones((self._num_true_parts * num_permutation ,) + patches.shape[2:])
        parts = np.ones((self._num_parts,) + patches[0].shape)
        d = np.prod(patches.shape[1:])
        #print(d,num_permutation)
        if 0:
            #\permutation = np.empty((num_permutation, num_permutation * d),dtype = np.int_)
            for a in range(num_permutation):
                if a == 0:
                    permutation[a] = np.arange(num_permutation * d)
                else:
                    permutation[a] = np.roll(permutation[a-1],d)
        flatpatches = patches.reshape((patches.shape[0],-1))
        print(flatpatches.shape)
        if 0:
            mm = BernoulliMM(n_components=num_parts, n_iter=20, tol=1e-15,n_init=2, random_state=0, min_prob=min_prob, verbose=False)
            print(mm.fit(flatpatches))
            print('AIC', mm.aic(flatpatches))
            print('BIC', mm.bic(flatpatches))
            #import pdb; pdb.set_trace()
            parts = mm.means_.reshape((num_parts,)+patches.shape[1:])
            #self._weights = mm.weights_
        elif 0:
            
            from pnet.bernoulli import em
            print("before EM")
            
            ret = em(flatpatches, self._num_true_parts,10,mu_truncation = min_prob, permutation = permutation, numpy_rng=self._settings.get('em_seed',0),verbose=True)
            comps = ret[3]
            parts = ret[1].reshape((self._num_true_parts * num_permutation,) + patches.shape[2:])
            self._weights = np.arange(self._num_parts)
        else:
            rng = np.random.RandomState(self._settings.get('em_seed',0))
            from pnet.latentShiftEM import LatentShiftEM
            #from latentShiftEM import latentShiftEM
            result = LatentShiftEM(flatpatches,num_mixture_component = self._num_parts, parts_shape = (self._part_shape[0],self._part_shape[1],8),region_shape = (self._sample_shape[1],self._sample_shape[1],8),shifting_shape = self._shifting_shape,max_num_iteration = 25, loglike_tolerance=1e-3, mu_truncation = (1,1),additional_mu = None, permutation = None, numpy_rng=rng, verbose = True)
            comps = result[3]
            print(comps.shape)
            print(original_patches.shape)
            print(result[1].shape)
            parts = result[1].reshape((self._num_parts,self._part_shape[0], self._part_shape[1],8))
            self._bkg_probability = result[4]
        self._parts = parts
        print(comps[:50,0])
        print(comps[:50,1])
        self._visparts = np.asarray([original_patches[comps[:,0]==k,comps[comps[:,0]==k][:,1]].mean(0) for k in range(self._num_parts)])
        print(self._visparts.shape)
        import amitgroup.plot as gr
        gr.images(self._visparts, zero_to_one=False, show=False,vmin=0,vmax = 1, fileName = 'moduleShiftingParts1.png')        
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
            w, h = [Xi.shape[i]-self._sample_shape[i]+1 for i in range(2)]

            # TODO: Maybe shuffle an iterator of the indices?
            indices = list(itr.product(range(w-1), range(h-1)))
            rs.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            for sample in range(samples_per_image):
                N = 200
                flag = 0
                for tries in range(N):
                    x, y = next(i_iter)
                    selection = [slice(x, x+self._sample_shape[0]), slice(y, y+self._sample_shape[1])]

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
                            #return np.asarray(patches),np.asarray(patches_original)
                            flag = 1
                        break
                if(flag ==1):
                    break
                    if tries == N-1:
                        ag.info('WARNING: {} tries'.format(N))
        patches = np.asarray(patches)
        patches_original = np.asarray(patches_original)
        newPatches = []
        newPatches_original = []
        if 0:
            print(patches.shape)
            print(patches_original.shape)
            for i in range(self._shifting_shape[0]):
                for j in range(self._shifting_shape[1]):
                    newPatches.append(patches[:,i:i+self._part_shape[0],j:j+self._part_shape[1],:])
                    newPatches_original.append(patches_original[:,i:i+self._part_shape[0],j:j+self._part_shape[1]])
            print(len(newPatches))
            print(len(newPatches_original))
            for i in range(9):
                print(newPatches[i].shape)
            patches = np.asarray(newPatches)
            patches_original = np.asarray(newPatches_original)
            patches = np.swapaxes(patches,0,1)
            patches_original = np.swapaxes(patches_original,0,1)
        else:
            for i in range(self._shifting_shape[0]):
                for j in range(self._shifting_shape[1]):
                    newPatches_original.append(patches_original[:,i:i+self._part_shape[0], j:j+self._part_shape[1]])
            patches_original = np.asarray(newPatches_original)
            patches_original = np.swapaxes(patches_original,0,1)
        print(patches.shape)
        print(patches_original.shape)
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
        d['shifting_shape'] = self._shifting_shape
        d['visparts'] = self._visparts
        d['settings'] = self._settings
        d['parts'] = self._parts
        d['weights'] = self._weights
        d['bkg_probability'] = self._bkg_probability
        d['sample_shape'] = self._sample_shape
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'],d['part_shape'], d['shifting_shape'],settings=d['settings'])
        obj._bkg_probability = d['bkg_probability']
        obj._parts = d['parts']
        obj._weights = d['weights']
        obj._visparts = d['visparts']
        obj._sample_shape = d['sample_shape']
        return obj
