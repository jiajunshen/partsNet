from __future__ import division, print_function,absolute_import
import pylab as plt
import amitgroup.plot as gr
import numpy as np
import amitgroup as ag
import os
import pnet
import matplotlib.pylab as plot

def extract(ims,allLayers):
    print(allLayers)
    curX = ims
    for layer in allLayers:
        curX = layer.extract(curX)
    return curX

# load the trained Image
#X =  np.load('testJun17NonSequential.npy')
X = np.load('Jun26Rot.npy')
model = X.item()
# get the parts Layer
numParts1 = model['layers'][0]['num_true_parts'] * model['layers'][0]['num_orientations']
patchShape = (6,6)
net = pnet.PartsNet.load_from_dict(model)
allLayer = net.layers
print(allLayer)
ims,labels = ag.io.load_mnist('training') 
extractedFeature = []

extractedFeature = allLayer[0].extract(ims[0:1000])[0]

extractedParts1 = extractedFeature
print(extractedParts1.shape)
partsPlot1 = np.zeros((numParts1,) + patchShape)
partsCodedNumber1 = np.zeros(numParts1)

for i in range(1000):
    codeParts1 = extractedParts1[i].reshape(extractedParts1[i].shape[0:2])
    for m in range(28 - patchShape[0] + 1):
        for n in range(28 - patchShape[0] + 1):
            if(codeParts1[m,n]!=-1):
                partsPlot1[codeParts1[m,n]]+=ims[i,m:m+patchShape[0],n:n+patchShape[0]] 
                partsCodedNumber1[codeParts1[m,n]]+=1
for j in range(numParts1):
    partsPlot1[j] = partsPlot1[j]/partsCodedNumber1[j]
 
#fileString = 'firstLayerPartsNonSequencial.png'
fileString = 'firstLayeriRotNonSe.png'
gr.images(partsPlot1,zero_to_one=False, show=False,vmin = 0, vmax = 1, fileName = fileString) 
#print(partsPlot1.shape)
#gr.images(partsPlot1[0:200],vmin = 0,vmax = 1)
#gr.images(partsPlot2,vmin = 0,vmax = 1)
#print(partsCodedNumber1)
print("-----------------")
#print(partsCodedNumber2)
#return partsPlot1,partsPlot2,extractedFeature
