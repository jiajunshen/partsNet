import amitgroup as ag
import pnet
import numpy as np
edgeLayer = pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05)
digits = np.arange(10)
data = ag.io.load_mnist("training",selection = slice(10000),return_labels = False)
edgeData = edgeLayer.extract(data)
partsLayer = pnet.PartsLayer(100, (8, 8), settings=dict(outer_frame=0, 
                                          threshold=40, 
                                          samples_per_image=40, 
                                          max_samples=100000, 
                                          min_prob=0.005))
allPatches = partsLayer._get_patches(edgeData,edgeData[:,:,:,0])
allPatches = allPatches[0]
allPatches = allPatches.reshape(allPatches.shape[0], -1)
from latentShiftEM import LatentShiftEM
rng = np.random.RandomState()

from pnet.bernoulli import em
#ret = em(allPatches, 50,20,verbose=True)
em = LatentShiftEM(allPatches, num_mixture_component = 50, parts_shape = (6,6,8), region_shape = (8,8,8), shifting_shape= (3,3), max_num_iteration = 25,loglike_tolerance=1e-3, mu_truncation = (1, 1), additional_mu = None, permutation = None, numpy_rng=rng, verbose = True)
from latentShiftEM import Extract
result = Extract(allPatches,num_mixture_component = 50, parts_shape = (6,6,8), region_shape = (8,8,8), shifting_shape = (3,3), mu = em[1],bkg_probability = em[4])
print(result)
print(np.mean(result[:,0] == em[3][:,0]))
print(result[:,0] == em[3][:,0])
print(np.where(result[:,0]!=em[3][:0]))
print(em[3])
