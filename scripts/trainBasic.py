from __future__ import division, print_function, absolute_import
__author__ = 'jiajunshen'
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


layers = [
    pnet.NormalizeLayer(),
    #pnet.IntensityThresholdLayer(),

    pnet.RandomPartitionSVMLayer(num_parts=32, part_shape=(5,5),settings=dict(outer_frame=2,
                                                                                threshold = 0,
                                                                                sample_per_image=500,
                                                                                max_samples=10000000,
                                                                                patch_extraction_seed = 0
    )),
    #pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),
    #pnet.IntensityThresholdLayer(),
    #pnet.PartsLayer(100, (6, 6), settings=dict(outer_frame=0,
    #                                          threshold=40,
    #                                          samples_per_image=50,
    #                                          max_samples=5000,
    #                                          min_prob=0.005,
    #                                          )),
    pnet.MaxPoolingLayer(shape=(2,2), strides=(2, 2)),
    pnet.RandomPartitionSVMLayer(num_parts=32, part_shape=(5,5),settings=dict(outer_frame=2,
                                                                                threshold = 0,
                                                                                sample_per_image=100,
                                                                                max_samples=5000000,
                                                                                patch_extraction_seed = 0
    )),
    pnet.MaxPoolingLayer(shape=(2,2), strides=(2, 2)),
    pnet.RandomPartitionSVMLayer(num_parts=64, part_shape=(5,5),settings=dict(outer_frame=2,
                                                                                threshold = 0,
                                                                                sample_per_image=50,
                                                                                max_samples=250000,
                                                                                patch_extraction_seed = 0
    )),
    pnet.MaxPoolingLayer(shape=(2,2), strides=(2, 2)),
    #pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001,block_size=200)
    pnet.SVMClassificationLayer(C=None)
]

net = pnet.PartsNet(layers)

digits = range(10)
#ims = ag.io.load_mnist('training', selection=slice(10000), return_labels=False)
#label = None
ims,label = ag.io.load_cifar10('training', selection=slice(50000))

#print(net.sizes(X[[0]]))

net.train(ims,label)

"""
sup_ims = []
sup_labels = []
# Load supervised training data
for d in digits:
    ims0 = ag.io.load_mnist('training', [d], selection=slice(10), return_labels=False)
    sup_ims.append(ims0)
    sup_labels.append(d * np.ones(len(ims0), dtype=np.int64))

sup_ims = np.concatenate(sup_ims, axis=0)
sup_labels = np.concatenate(sup_labels, axis=0)

net.train(sup_ims, sup_labels)
"""

net.save(args.model)

if args.log:
    net.infoplot(vz)
    vz.finalize()
