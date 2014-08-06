from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet

@Layer.register('moduloShifting_Rotation_Parts_layer')
class ModuloShiftingRotationPartsLayer(Layer):
    def __init__(self, num_parts, part_shape,shifting_shape,num_rot, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._shifting_shape = shifting_shape
        self._settings = settings
        self._num_rot = num_rot
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
        th = 5
        n_coded = settings.get('n_coded', 1)
        print(th) 
        support_mask = np.ones(part_shape, dtype=np.bool)

        from pnet.cyfuncs import code_index_map_general
        #print("=================")
        #print(x.shape)
        #print("=================")
        print(np.mean(parts == self._parts))
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

    def extract_without_edges(self,X):

        assert self._parts is not None, "Must be trained before calling extract"
        assert self._bkg_probability is not None, "bkg probability is None"
        from pnet.latentShiftRotationEM import Extract
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
            
            
            
            result = Extract(data = x_patches, num_mixture_component = self._num_parts,num_rot = self._num_rot, parts_shape = (self._part_shape[0],self._part_shape[1],8),region_shape = (self._sample_shape[1],self._sample_shape[1],8),shifting_shape = self._shifting_shape, mu = self._parts.reshape(self._num_parts * self._num_rot,-1),bkg_probability = self._bkg_probability)[:,0]
            result = result.reshape((x_patches_shape[0],x_patches_shape[1]))
            print(result.shape)
            result = result.reshape((x_patches_shape[0], (dataShape[1] - self._sample_shape[0] + 1) , (dataShape[2] - self._sample_shape[1] + 1), 1))
            print(result.shape)
            finalResult[p*block_size:(p+1)*block_size] = result
        finalResult = np.array(finalResult, dtype = np.int64)
        return finalResult, self._num_parts * self._num_rot

    def extract_no_edge(self,X):
        assert self._parts is not None, "Must be trained before calling extract"
        
        bedges_settings = self._settings['bedges']
        radius = bedges_settings['radius']
        #X = ag.features.bspread(X, spread=self._settings['bedges']['spread'], radius = self._settings['bedges']['radius'])

        support_mask = np.ones(self._part_shape, dtype=np.bool)
        th = self._settings['threshold']
        n_coded = 1
        from pnet.cyfuncs import code_index_map_general
        feature_map = code_index_map_general(X, self._parts, support_mask.astype(np.uint8),th,outer_frame=self._settings['outer_frame'],n_coded = 1, standardize=self._settings.get('standardize',0)
        min_percentile = self._settings.get('min_percentile',0.0))
        
        # Rotation spreading?
        rotspread = self._settings.get('rotation_spreading_radius', 0)
        if rotspread > 0:
            between_feature_spreading = np.zeros((self._num_parts * self._num_rot , rotspread*2 + 1), dtype=np.int64)
            ORI = self._num_rot

            for f in range(self._num_parts * self._num_rot):
                thepart = f // ORI
                ori = f % ORI 
                for i in range(rotspread*2 + 1):
                    between_feature_spreading[f,i] = thepart * ORI + (ori - rotspread + i) % ORI

            bb = np.concatenate([between_feature_spreading, -np.ones((1, rotspread*2 + 1), dtype=np.int64)], 0)

            # Update feature map
            feature_map = bb[feature_map[...,0]]
        
        return feature_map

    def extract(self,X):
        assert self._parts is not None, "Must be trained before calling extract"
        
        bedges_settings = self._settings['bedges']
        radius = bedges_settings['radius']
        X = self._extract_many_edges(bedges_settings, self._settings, X, must_preserve_size = True)
        X = ag.features.bspread(X, spread=self._settings['bedges']['spread'], radius = self._settings['bedges']['radius'])

        support_mask = np.ones(self._part_shape, dtype=np.bool)
        th = self._settings['threshold']
        n_coded = 1
        from pnet.cyfuncs import code_index_map_general
        feature_map = code_index_map_general(X, self._parts, support_mask.astype(np.uint8),th,outer_frame=self._settings['outer_frame'],n_coded = 1, standardize=self._settings.get('standardize',0),
        min_percentile = self._settings.get('min_percentile',0.0))
        
        # Rotation spreading?
        rotspread = self._settings.get('rotation_spreading_radius', 0)
        if rotspread > 0:
            between_feature_spreading = np.zeros((self._num_parts * self._num_rot , rotspread*2 + 1), dtype=np.int64)
            ORI = self._num_rot

            for f in range(self._num_parts * self._num_rot):
                thepart = f // ORI
                ori = f % ORI 
                for i in range(rotspread*2 + 1):
                    between_feature_spreading[f,i] = thepart * ORI + (ori - rotspread + i) % ORI

            bb = np.concatenate([between_feature_spreading, -np.ones((1, rotspread*2 + 1), dtype=np.int64)], 0)

            # Update feature map
            feature_map = bb[feature_map[...,0]]
        
        return feature_map

    def extract_old(self,X):
        print("+++++++++++++++++++++++++++++++++++++++++")
        assert self._parts is not None, "Must be trained before calling extract"
        assert self._bkg_probability is not None, "bkg probability is None"
        from pnet.latentShiftRotationEM import Extract
        bedges_settings = self._settings['bedges']
        radius = bedges_settings['radius']
        X = self._extract_many_edges(bedges_settings,self._settings,X, must_preserve_size=True)
        
        dataShape = X.shape

        print(X.shape)
        block_size = 100
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
            
            
            
            result = Extract(data = x_patches, num_mixture_component = self._num_parts,num_rot = self._num_rot, parts_shape = (self._part_shape[0],self._part_shape[1],8),region_shape = (self._sample_shape[1],self._sample_shape[1],8),shifting_shape = self._shifting_shape, mu = self._parts.reshape(self._num_parts * self._num_rot,-1),bkg_probability = self._bkg_probability)[:,0]
            result = result.reshape((x_patches_shape[0],x_patches_shape[1]))
            print(result.shape)
            result = result.reshape((x_patches_shape[0], (dataShape[1] - self._sample_shape[0] + 1) , (dataShape[2] - self._sample_shape[1] + 1), 1))
            print(result.shape)
            finalResult[p*block_size:(p+1)*block_size] = result
        finalResult = np.array(finalResult, dtype = np.int64)
        return finalResult, self._num_parts * self._num_rot


    def extract_without_edges_batch(self,X):

        assert self._parts is not None, "Must be trained before calling extract"
        from pnet.latentShiftRotationEM import Extract
        X = np.array(X, dtype = np.uint8)
        b = int(X.shape[0] // 400)     
        print(X.shape) 
        sett = (self._settings, self._num_parts, self._num_rot, self._part_shape,self._parts)
        if 1:
            feat = self._extract_batch(X, *sett)
        else:
            X_batches = np.array_split(X, b)
            args = ((X_b,) + sett for X_b in X_batches)
            feat = np.concatenate([batch for batch in pnet.parallel.starmap_unordered(self._extract_batch, args)])
        return (feat, self._num_parts * self._num_rot)
    
    
    @property
    def trained(self):
        return self._parts is not None 

    def train(self, X, Y=None, OriginalX = None):
        assert Y is None
        ag.info('Extracting patches')
        print("Begin")
        patches, patches_original = self._get_patches(X,OriginalX)
        
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches, patches_original)

        

    def train_from_samples(self, patches, original_patches):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        print(patches.shape)
        print(original_patches.shape)
        from pnet.bernoullimm import BernoulliMM
        print(patches.shape)
        min_prob = self._settings.get('min_prob', 0.01)
        #num_permutation = self._shifting_shape[0] * self._shifting_shape[1]
        #parts = np.ones((self._num_true_parts * num_permutation ,) + patches.shape[2:])
        parts = np.ones((self._num_parts,) + patches[0].shape)
        d = np.prod(patches.shape[2:])
        flatpatches = patches.reshape((patches.shape[0], -1))
        rng = np.random.RandomState(self._settings.get('em_seed',0))
        from pnet.latentShiftRotationEM import LatentShiftRotationEM
        #from latentShiftEM import latentShiftEM
        permutation = np.empty((self._num_rot, self._num_rot * d),dtype = np.int_)
        for a in range(self._num_rot):
            if a == 0:
                permutation[a] = np.arange(self._num_rot * d)
            else:
                permutation[a] = np.roll(permutation[a-1],d)
        
        partsPermutation = np.empty((self._num_rot, self._num_rot * self._part_shape[0] * self._part_shape[1] * 8),dtype = np.int_)
        for a in range(self._num_rot):
            if a == 0:
                partsPermutation[a] = np.arange(self._num_rot * self._part_shape[0] * self._part_shape[1] * 8)
            else:
                partsPermutation[a] = np.roll(partsPermutation[a-1],self._part_shape[0] * self._part_shape[1] * 8)
        
        
        
        result = LatentShiftRotationEM(flatpatches,num_mixture_component = self._num_parts, parts_shape = (self._part_shape[0],self._part_shape[1],8),region_shape = (self._sample_shape[1],self._sample_shape[1],8),shifting_shape = self._shifting_shape,num_rot = self._num_rot, max_num_iteration = 25, loglike_tolerance=1e-3, mu_truncation = (1,1),additional_mu = None, permutation = permutation,partPermutation = partsPermutation, numpy_rng=rng, verbose = True)
        comps = result[3]
        print(comps.shape)
        print(original_patches.shape)
        print(result[1].shape)
        parts = result[1].reshape((self._num_parts * self._num_rot,self._part_shape[0], self._part_shape[1],8))
        self._bkg_probability = result[4]
        self._parts = parts
        print(original_patches.shape)
        print(comps[:50,0])
        print(comps[:50,1])
        print(comps[50:150,2])
        allShift = np.zeros(25)
        for i in comps[:,2]:
            allShift[i]+=1
        print(allShift/np.sum(allShift))
        print(np.mean(comps[:,2]==4))
        print(comps.shape)
        print(comps[comps[:0] == 1][:1].shape)
        print(original_patches.shape)
        original_patches = np.swapaxes(original_patches,0,1)
        print(comps[:,0]==1)
        print(comps[comps[:,0]==1][:,1])
        print(comps[comps[:,0]==1][:,2])
        self._visparts = np.asarray([original_patches[comps[:,0]==k, comps[comps[:,0]==k][:,1],comps[comps[:,0]==k][:,2]].mean(0) for k in range(self._num_parts)])
        print(self._visparts.shape)
        import amitgroup.plot as gr
        #gr.images(original_patches[:5,:,:,:,:].reshape((-1,) + self._part_shape),zero_to_one=False,show=False,vmin = 0,vmax = 1, fileName = 'testImg.png')
        gr.images(self._visparts, zero_to_one=False, show=False,vmin=0,vmax = 1, fileName = 'moduleShiftingParts1.png')        
        return parts

    def _get_patches(self, X, X_original = None):
        bedges_settings = self._settings['bedges']
        samples_per_image = self._settings['samples_per_image']
        fr = self._settings.get('outer_frame', 0)
        the_patches = []
        the_originals = []
        ag.info("Extracting patches from")
        #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True, lastaxis=True)
        setts = self._settings['bedges'].copy()
        radius = setts['radius']
        setts2 = setts.copy()
        setts['radius'] = 0
        ps = self._part_shape


        ORI = self._num_rot
        POL = self._settings.get('polarities', 1)
        assert POL in (1, 2), "Polarities must be 1 or 2"
        #assert ORI%2 == 0, "Orientations must be even, so that opposite can be collapsed"

        # LEAVE-BEHIND

        from skimage import transform

        for n, img in enumerate(X):
            size = img.shape[:2]
            # Make it square, to accommodate all types of rotations
            new_size = int(np.max(size) * np.sqrt(2))
            img_padded = ag.util.pad_to_size(img, (new_size, new_size))
            pad = [(new_size-size[i])//2 for i in range(2)]

            angles = np.arange(0, 360, 360/ORI)
            radians = angles*np.pi/180
            all_img = np.asarray([transform.rotate(img_padded, angle, resize=False, mode='nearest') for angle in angles])
            # Add inverted polarity too
            if POL == 2:
                all_img = np.concatenate([all_img, 1-all_img])


            # Set up matrices that will translate a position in the canonical image to
            # the rotated iamges. This way, we're not rotating each patch on demand, which
            # will end up slower.
            matrices = [pnet.matrix.translation(new_size/2, new_size/2) * pnet.matrix.rotation(a) * pnet.matrix.translation(-new_size/2, -new_size/2) for a in radians]

            # Add matrices for the polarity flips too, if applicable
            matrices *= POL 

            #inv_img = 1 - img
            all_unspread_edges = self._extract_many_edges(bedges_settings, self._settings, all_img, must_preserve_size=True)

            # Spread the edges
            all_edges = ag.features.bspread(all_unspread_edges, spread=setts['spread'], radius=radius)
            #inv_edges = ag.features.bspread(inv_unspread_edges, spread=setts['spread'], radius=radius)

            # How many patches could we extract?

            # This avoids hitting the outside of patches, even after rotating.
            # The 15 here is fairly arbitrary
            #avoid_edge = int(0 + np.max(ps)*np.sqrt(2))
            avoid_edge = int(np.ceil(np.max(self._sample_shape) * (np.sqrt(2) - 1) / (2 * np.sqrt(2))))
            # This step assumes that the img -> edge process does not down-sample any

            # TODO: Maybe shuffle an iterator of the indices?

            # These indices represent the center of patches
            indices = list(itr.product(range(pad[0]+avoid_edge, pad[0]+img.shape[0]-avoid_edge), range(pad[1]+avoid_edge, pad[1]+img.shape[1]-avoid_edge)))

            #print(indices," ", pad[0]," ", pad[1]," ",avoid_edge)
            import random
            random.seed(0)
            random.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            minus_ps = [-ps[i]//2 for i in range(2)]
            plus_ps = [minus_ps[i] + ps[i] for i in range(2)]

            E = all_edges.shape[-1]
            #th = _threshold_in_counts(self._settings, E, self._settings['bedges']['contrast_insensitive'], self._part_shape)
            th = self._settings['threshold']

            max_samples = self._settings.get('max_samples', np.inf)

            rs = np.random.RandomState(0)

            rs2 = np.random.RandomState(0)

            for sample in range(samples_per_image):
                flag= 0
                for tries in range(100):
                    for shiftX in range(self._shifting_shape[0]):
                        for shiftY in range(self._shifting_shape[1]):
                            #selection = [slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1])]
                            #x, y = random.randint(0, w-1), random.randint(0, h-1)
                            x, y = next(i_iter)

                            selection0 = [0, slice(x+minus_ps[0]-shiftX, x+plus_ps[0]+self._shifting_shape[0]-1-shiftX), slice(y+minus_ps[1]-shiftY, y+plus_ps[1] + self._shifting_shape[1] - 1 - shiftY)]
                            #selection0 = [0, slice(x, x+self._sample_shape[0]), slice(y, y+self._sample_shape[1])]
                            # Return grayscale patch and edges patch
                            unspread_edgepatch = all_unspread_edges[selection0]
                            edgepatch = all_edges[selection0]
                            #inv_edgepatch = inv_edges[selection]

                            #amppatch = amps[selection]
                            #edgepatch2 = edges2[selection]
                            #inv_edgepatch2 = inv_edges2[selection]
                            #edgepatch_nospread = edges_nospread[selection]

                            # TODO
                            unspread_edgepatch = edgepatch

                            # The inv edges could be incorproated here, but it shouldn't be that different.
                            if fr == 0:
                                tot = unspread_edgepatch.sum()
                            else:
                                tot = unspread_edgepatch[fr:-fr,fr:-fr].sum()


                            #if self.settings['threshold'] <= avg <= self.settings.get('max_threshold', np.inf): 
                            if th <= tot:
                                XY = np.matrix([x, y, 1]).T
                                # Now, let's explore all orientations

                                patch = np.zeros((ORI * POL,) + self._sample_shape + (E,))
                                vispatch = np.zeros((ORI * POL,) + self._sample_shape)
                                outRangeFlag = 0
                                for ori in range(ORI * POL):
                                    p = matrices[ori] * XY
                                    ip = [int(round(float(p[i]))) for i in range(2)]
                                    selection = [ori, slice(ip[0]+minus_ps[0] - shiftX, ip[0]+plus_ps[0] + self._shifting_shape[0] - 1 - shiftX), slice(ip[1]+minus_ps[1]-shiftY , ip[1]+plus_ps[1] + self._shifting_shape[1] - 1 - shiftY)]
                                    if patch[ori].shape == all_edges[selection].shape:
                                        patch[ori] = all_edges[selection]
                                        orig = all_img[selection]
                                        span = orig.min(), orig.max() 
                                        if span[1] - span[0] > 0:
                                            orig = (orig-span[0])/(span[1]-span[0])
                                        vispatch[ori] = orig 
                                    else:
                                        outRangeFlag = 1
                                        break 


                                # Randomly rotate this patch, so that we don't bias 
                                # the unrotated (and possibly unblurred) image

                                #shift = rs.randint(ORI)

                                #patch[:ORI] = np.roll(patch[:ORI], shift, axis=0)
                                #vispatch[:ORI] = np.roll(vispatch[:ORI], shift, axis=0)
                                if outRangeFlag==1:
                                    break
                                the_patches.append(patch)
                                the_originals.append(vispatch)

                                if len(the_patches) >= max_samples:
                                    flag = 1

                                    the_originals = np.asarray(the_originals)
                                    print(the_originals.shape)
                                    newPatches_original = []
                                    for i in range(self._shifting_shape[0]):
                                        for j in range(self._shifting_shape[1]):
                                            newPatches_original.append(the_originals[:,:,i:i+self._part_shape[0], j:j+self._part_shape[1]])
                                    the_originals = np.asarray(newPatches_original)
                                    the_originals = np.swapaxes(the_originals,0,2)
                                    print("__________________HERE___________________")
                                    print(the_originals.shape)
                                    return np.asarray(the_patches), np.asarray(the_originals)

                                break
                        if(flag == 1):
                            break
                            if tries == 99:
                                    print("100 tries!")
        the_originals = np.asarray(the_originals)
        print(the_originals.shape)
        newPatches_original = []
        for i in range(self._shifting_shape[0]):
            for j in range(self._shifting_shape[1]):
                newPatches_original.append(the_originals[:,:,i:i+self._part_shape[0], j:j+self._part_shape[1]])
        the_originals = np.asarray(newPatches_original)
        the_originals = np.swapaxes(the_originals,0,2)
        print("__________________HERE___________________")
        print(the_originals.shape)
        return np.asarray(the_patches), np.asarray(the_originals)

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
        d['num_rot'] = self._num_rot
        d['visparts'] = self._visparts
        d['settings'] = self._settings
        d['parts'] = self._parts
        d['weights'] = self._weights
        d['bkg_probability'] = self._bkg_probability
        d['sample_shape'] = self._sample_shape
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'],d['part_shape'], d['shifting_shape'],d['num_rot'],settings=d['settings'])
        obj._bkg_probability = d['bkg_probability']
        obj._parts = d['parts']
        obj._weights = d['weights']
        obj._visparts = d['visparts']
        obj._sample_shape = d['sample_shape']
        return obj
