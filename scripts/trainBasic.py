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

#net_pretrain = pnet.PartsNet.load("./test_360_scaled.npy")

print("before train")
layers = [
    pnet.NormalizeLayer(),
    #pnet.IntensityThresholdLayer(),

    #pnet.OrientedGaussianPartsLayer(32,1,(5,5),settings=dict(
    #    seed=0,
    #   n_init = 2,
    #    samples_per_image=10,
    #    max_samples=200000,
    #    channel_mode='together',
    #    coding="soft"
    #    #covariance_type = ''
    #)),
    #pnet.MaxPoolingLayer(shape=(2,2), strides=(2, 2)),

    pnet.RandomPartitionSVMLayer(num_parts=360, part_shape=(3,3),settings=dict(outer_frame=2,
                                                                                threshold = 0,
                                                                                samples_per_image=10,
                                                                                max_samples=500000,
                                                                                patch_extraction_seed = 17389,
                                                                                all_locations = False,
                                                                                use_theano = True,
                                                                                random_seeds = 17389
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
    pnet.RandomPartitionSVMLayer(num_parts=360, part_shape=(3,3),settings=dict(outer_frame=2,
                                                                                threshold = 0,
                                                                                samples_per_image=30,
                                                                                max_samples=100000,
                                                                                patch_extraction_seed = 37813,
                                                                                all_locations = False,
                                                                                use_theano = True,
                                                                                random_seeds = 37813
    )),
    pnet.MaxPoolingLayer(shape=(2,2), strides=(2, 2)),
    pnet.RandomPartitionSVMLayer(num_parts=360, part_shape=(3,3),settings=dict(outer_frame=2,
                                                                                threshold = 0,
                                                                                samples_per_image=30,
                                                                                max_samples=100000,
                                                                                patch_extraction_seed = 59359,
                                                                                all_locations = False,
                                                                                random_seeds = 59359
    )),
    pnet.MaxPoolingLayer(shape=(2,2), strides=(2, 2)),
    #pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001,block_size=200)
    pnet.SVMClassificationLayer(C=0.001)
]


#layers = net_pretrain.layers[:7]

#layers += [pnet.SGDSVMClassificationLayer()]

net = pnet.PartsNet(layers)

digits = range(10)
#ims = ag.io.load_mnist('training', selection=slice(10000), return_labels=False)
#label = None
ims,label = ag.io.load_cifar10('training', selection=slice(5000))

#print(net.sizes(X[[0]]))
print("start train")
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


