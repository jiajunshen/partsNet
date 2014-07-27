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

training_seed = 1

layers = [
    #pnet.IntensityThresholdLayer(),
    pnet.OrientedPartsLayer(50,8,(6,6),settings=dict(outer_frame = 2,#1
                                em_seed=training_seed,
                                n_init = 2,
                                threshold = 5,#2
                                samples_per_image=40,
                                max_samples=10000,
                                train_limit=20000,
                                rotation_spreading_radius = 0,
                                min_prob= 0.0005,
                                bedges = dict(k = 5,
                                               minimum_contrast = 0.05,
                                               spread = 'orthogonal',
                                                radius = 1,
                                                contrast_insensitive=False,
                                               ),
    )), 
    
    pnet.PoolingLayer(shape=(7, 7), strides=(1, 1)),
    #pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001),
    #pnet.SVMClassificationLayer(C=1.0),
    #pnet.PartsLayer(1000, (1, 1), settings=dict(outer_frame=0,
    #                                           threshold=5,
    #                                           samples_per_image=200,
    #                                           max_samples=200000,
    #                                           min_prob=0.0005,
    #                                          min_llh=-40,
    #                                          n_coded=2
    #                                              )),
    #pnet.PoolingLayer(shape=(2,2), strides=(2, 2)),
    
    #pnet.PartsLayer(200, (1, 1), settings=dict(outer_frame=0,
    #                                           threshold=5,
    #                                           samples_per_image=200,
    #                                           max_samples=200000,
    #                                           min_prob=0.0005,
    #                                          min_llh=-40,
    #                                          n_coded=2
    #                                          )),
    #pnet.PoolingLayer(shape=(10,10), strides=(10, 10)),
    #pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001,block_size=200),
    #pnet.SVMClassificationLayer(C=1.0)
]

net = pnet.PartsNet(layers)

digits = range(10)
ims = ag.io.load_mnist('training', selection=slice(10000), return_labels=False)

#print(net.sizes(X[[0]]))

net.train(ims)


net.save(args.model)

if args.log:
    net.infoplot(vz)
    vz.finalize()
