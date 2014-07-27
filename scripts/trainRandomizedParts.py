from __future__ import division, print_function, absolute_import 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', metavar='<parts net file>', type=argparse.FileType('wb'), help='Filename of model file')
parser.add_argument('--log', action='store_true')

args = parser.parse_args()

if args.log:
    from pnet.vzlog import default as vz
import numpy as np
import amitgroup as ag
import sys
ag.set_verbose(True)

from pnet.bernoullimm import BernoulliMM
import pnet
from sklearn.svm import LinearSVC 
import numpy as  np

layers = [
    #pnet.IntensityThresholdLayer(),
    
    pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),
    #pnet.IntensityThresholdLayer(),
    pnet.PartsLayer(100, (5, 5), settings=dict(outer_frame=0, 
                                              threshold=40, 
                                              samples_per_image=40, 
                                              max_samples=10000, 
                                              min_prob=0.005)),
    pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
    #    #pnet.SVMClassificationLayer(C=1.0),
    #pnet.PartsLayer(2000, (1, 1), settings=dict(outer_frame=0,
    #                                          threshold=6,
    #                                          samples_per_image=40,
    #                                          max_samples=10000,
    #                                          min_prob=0.0005,
    ##                                          min_llh=-40,
    #                                          n_coded=2
    #                                          )),
    #pnet.PoolingLayer(shape=(3,3), strides=(3, 3)),
    pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001),

    #pnet.SVMClassificationLayer(C=1.0)
]

min_prob = 0.005
layers[1]._parts = np.random.uniform(min_prob,1-min_prob,(100,5,5,8))
layers[1]._weights = (np.array(np.random.multinomial(900,[1/100.]*100),dtype = float) + 1)/1000
print(layers[1]._weights)
net = pnet.PartsNet(layers)

digits = range(10)
ims = ag.io.load_mnist('training', selection=slice(10000), return_labels=False)

#print(net.sizes(X[[0]]))

net.train(ims)

sup_ims = []
sup_labels = []
# Load supervised training data
for d in digits:
    ims0 = ag.io.load_mnist('training', [d], selection=slice(1000), return_labels=False)
    sup_ims.append(ims0)
    sup_labels.append(d * np.ones(len(ims0), dtype=np.int64))

sup_ims = np.concatenate(sup_ims, axis=0)
sup_labels = np.concatenate(sup_labels, axis=0)

net.train(sup_ims, sup_labels)


net.save(args.model)

if args.log:
    net.infoplot(vz)
    vz.finalize()
