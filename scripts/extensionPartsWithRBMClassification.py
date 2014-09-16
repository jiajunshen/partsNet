import cPickle
import gzip
import time
import PIL.Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from __future__ import division, print_function,absolute_import
import pylab as plt
import amitgroup.plot as gr
import numpy as np
import amitgroup as ag
import os
import pnet
import matplotlib.pylab as plot
from pnet.cyfuncs import index_map_pooling
from Queue import Queue


def test(ims,labels,net):
    yhat = net.classify((ims,1000))
    return yhat == labels

newPooledExtract = np.array(pooledExtract[:1000]).reshape(1000,4,4,500)
for p in range(4):
    for q in range(4):
        location1 = newData[:,p,q,:]
        data = weights[p,q,:500,:]
        X = np.array(data.reshape(500,20),dtype=np.double)
        kmeans = sklearn.cluster.k_means(np.array(X,dtype = np.double),10)[1]
        skipIndex = np.argmax(np.bincount(kmeans))
        #Put in all the array of group index here
        groupIndexArray = [[] for m in range(10)]
        for i in range(10):
            if i == skipIndex:
                continue
            testIndex = i
            indexArray = np.where(kmeans == testIndex)[0]
            groupIndexArray[testIndex].append(indexArray)

        poolingIndex = [[] for m in range(500)]
        for k in np.where(np.max(location1,axis=0)!=0)[0]:
            if kmeans[k] == skipIndex:
                continue
            else:
                distanceArray = np.array([np.sum((X[m,:]-X[k,:]) * (X[m,:]-X[k,:])) for m in groupIndexArray[kmeans[k]][0]])
                #print(distanceArray.shape)
                numPooling = (distanceArray.shape[0] + 1)//2
#                 print(numPooling)
                finalPooling = groupIndexArray[kmeans[k]][0][np.argsort(distanceArray)[:numPooling]]
                #print(k, finalPooling)
                poolingIndex[k].append(finalPooling)
                
        for r in range(1000):
            print(r)
            for m in range(500):
                if newPooledExtract[r,p,q,m] == 1:
                    if size(poolingIndex[m])==0:
                        continue
                    else:
#                         print(poolingIndex[m][0])
                        newPooledExtract[r,p,q,:][poolingIndex[m][0]] = 1



testImg_curX = np.load('../thirdLevelCurx_Test.npy')
testImg_curX = np.array(testImg_curX, dtype = np.int64)
pooledTest = poolHelper.extract((testImg_curX[:,:,:,np.newaxis],500))
testImg,testLabels = ag.io.load_mnist('testing')


newPooledExtractTest = np.array(pooledTest[:10000]).reshape(10000,4,4,500)
for p in range(4):
    for q in range(4):
        location1 = newData[:,p,q,:]
        data = weights[p,q,:500,:]
        X = np.array(data.reshape(500,20),dtype=np.double)
        kmeans = sklearn.cluster.k_means(np.array(X,dtype = np.double),10)[1]
        skipIndex = np.argmax(np.bincount(kmeans))
        #Put in all the array of group index here
        groupIndexArray = [[] for m in range(10)]
        for i in range(10):
            if i == skipIndex:
                continue
            testIndex = i
            indexArray = np.where(kmeans == testIndex)[0]
            groupIndexArray[testIndex].append(indexArray)

        poolingIndex = [[] for m in range(500)]
        for k in np.where(np.max(location1,axis=0)!=0)[0]:
            if kmeans[k] == skipIndex:
                continue
            else:
                distanceArray = np.array([np.sum((X[m,:]-X[k,:]) * (X[m,:]-X[k,:])) for m in groupIndexArray[kmeans[k]][0]])
                #print(distanceArray.shape)
                numPooling = (distanceArray.shape[0] + 1)//2
#                 print(numPooling)
                finalPooling = groupIndexArray[kmeans[k]][0][np.argsort(distanceArray)[:numPooling]]
                #print(k, finalPooling)
                poolingIndex[k].append(finalPooling)
                
        for r in range(10000):
            print(r)
            for m in range(500):
                if newPooledExtractTest[r,p,q,m] == 1:
                    if size(poolingIndex[m])==0:
                        continue
                    else:
#                         print(poolingIndex[m][0])
                        newPooledExtractTest[r,p,q,:][poolingIndex[m][0]] = 1


#Train a class Model#

classificationLayers = [pnet.SVMClassificationLayer(C = 1.0)]
classificationNet = pent.PartsNet(classificationLayers)
classificationNet.train((np.array(newPooledExtract,dtype = np.int64),500, sup_labels[:100]))
print("Training Success!")
testImg_Input = np.array(newPooledExtractTest, dtype = np.int64)
testImg_batches = np.array_split(testImg_Input, 200)
testLabels_batches = np.array_split(testLabels, 200)

args = [tup + (classificationNet,) for tup in zip(testImg_batches, testLabels_batches)]

def format_error_rate(pr):
    return "{:.2f}%".format(100 * (1-pr))

print("Testing Starting...")

for i, res in enumerate(pnet.parallel.starmap_unordered(test,args)):
    if i !=0 and i % 20 ==0:
        print("{0:05}/{1:05} Error rate: {2}".format(total, len(ims),format_error_rate(pr)))

    corrects += res.sum()
    total += res.size

    pr = corrects / total

print("Final error rate:", format_error_rate(pr))



