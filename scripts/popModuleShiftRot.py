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
def extract(ims,allLayers):
    #print(allLayers)
    curX = ims
    for layer in allLayers:
        #print('-------------')
        #print(layer)
        curX = layer.extract(curX)
        #print(np.array(curX).shape)
        #print('------------------')
    return curX

def partsPool(originalPartsRegion, numParts):
    partsGrid = np.zeros((1,1,numParts))
    for i in range(originalPartsRegion.shape[0]):
        for j in range(originalPartsRegion.shape[1]):
            if(originalPartsRegion[i,j]!=-1):
                partsGrid[0,0,originalPartsRegion[i,j]] = 1
    return partsGrid



def test(ims,labels,net):
    yhat = net.classify((ims,800))
    return yhat == labels
    


#def trainPOP():
if pnet.parallel.main(__name__):
    #X = np.load("testMay151.npy")
    #X = np.load("_3_100*6*6_1000*1*1_Jun_16_danny.npy")
    #X = np.load("original6*6.npy")
    #X = np.load("sequential6*6.npy")
    #X = np.load("Jul22ModuloShiftRot.npy")
    X = np.load("Modulo_shift_Rot_p10s66r8ms55tp10000fr0th80.npy")
    model = X.item()
    # get num of Parts
    numParts = model['layers'][0]['num_parts']
    print(numParts,model)
    net = pnet.PartsNet.load_from_dict(model)
    allLayer = net.layers
    shiftRotationLayer = allLayer[0]
    ims,labels = ag.io.load_mnist('training')
    trainingDataNum = 100000
    firstLayerShape = 6
    secondLayerCodedNumber = 0
    secondLayerShape = 12
    codeShape = secondLayerShape - firstLayerShape + 1
    num_rot = shiftRotationLayer._num_rot
    part_shape = shiftRotationLayer._part_shape
    sample_shape = shiftRotationLayer._sample_shape
    largeModuleShiftRotationPartsLayer = pnet.OrientedPartsLayer(50, num_rot, (secondLayerShape, secondLayerShape), settings = dict(outer_frame = 2, em_seed = None, n_init = 2, threshold = 5, samples_per_image = 40, max_samples = 100000, train_limit = 100000, rotation_spread_radius = 0, min_prob = 0.0005, bedges = dict(k = 5, minimum_contrast = 0.05, spread = 'orthogonal', radius = 1, contrast_insensitive = False, )))
    #largeModuleShiftRotationPartsLayer = pnet.ModuloShiftingRotationPartsLayer(num_parts = 50, part_shape = (secondLayerShape - 2,secondLayerShape - 2),shifting_shape = (3,3),num_rot = num_rot, settings = shiftRotationLayer._settings) 
    patches, patches_original = largeModuleShiftRotationPartsLayer._get_patches(ims[0:10000])
    print(patches_original.shape)
    print(patches.shape)
    patches = patches[:trainingDataNum]
    print(patches.shape,shiftRotationLayer._parts.shape)
    #extractedFeature = shiftRotationLayer.extract_without_edges(patches.reshape((trainingDataNum * num_rot, secondLayerShape, secondLayerShape, 8)))[0]
    extractedFeature = shiftRotationLayer.extract_without_edges_batch(patches.reshape((trainingDataNum * num_rot, secondLayerShape, secondLayerShape, 8)))[0]
    print(extractedFeature.shape)
    #print(extractedFeature.reshape(trainingDataNum,4,7,7,1)[:20,:,2,2,0]) 
    permutation = np.empty((num_rot,num_rot * codeShape * codeShape),np.int_)
    for a in range(num_rot):
        if a == 0:
            permutation[a] = np.arange(num_rot * codeShape * codeShape)
        else:
            permutation[a] = np.roll(permutation[a-1], codeShape * codeShape)
    print(permutation)
    
    originalPartsPermutation = np.empty((num_rot,num_rot * secondLayerShape * secondLayerShape),np.int_)
    for a in range(num_rot):
        if a == 0:
            originalPartsPermutation[a] = np.arange(num_rot * secondLayerShape * secondLayerShape)
        else:
            originalPartsPermutation[a] = np.roll(originalPartsPermutation[a-1], secondLayerShape * secondLayerShape)
    print(originalPartsPermutation)
    
     
    patches_original = patches_original.reshape(trainingDataNum, -1)
    extractedFeature = extractedFeature.reshape(trainingDataNum,-1)
    
    for i in range(trainingDataNum):
        centerCode = extractedFeature[i].reshape(num_rot,codeShape,codeShape)[0,(codeShape - 1)/2, (codeShape - 1)/2]
        #print(centerCode)
        if centerCode == -1:
            continue
        if 0:
            permutationRot = -1
            for j in range(num_rot):
                if centerCode[j]%num_rot == 0:
                    permutationRot = j
                    break
            if(permutationRot == -1):
                break
        else:
            permutationRot = centerCode%4
        extractedFeature[i] = extractedFeature[i,permutation[permutationRot]]
        extractedFeature[i,int((codeShape * codeShape - 1)/2)] = centerCode - permutationRot
        #print(centerCode,permutationRot, extractedFeature[i].reshape(num_rot,codeShape,codeShape)[0, (codeShape - 1)/2, (codeShape -1) / 2])
        patches_original[i] = patches_original[i, originalPartsPermutation[permutationRot]]
   
    partsRegion = [[] for x in range(int(numParts))]
    originalPartsList = [[] for x in range(int(numParts))]
    for i in range(trainingDataNum):
        centerCode = extractedFeature[i].reshape(num_rot,codeShape,codeShape)[0,(codeShape - 1)/2, (codeShape - 1)/2]
        if centerCode != -1:
            if(centerCode%num_rot!=0):
                print('what')
            partsGrid = np.array([partsPool(extractedFeature[i].reshape(num_rot,codeShape,codeShape)[j,:,:],numParts*num_rot) for j in range(num_rot)])
            partsRegion[centerCode//num_rot].append(partsGrid)
            originalPartsList[centerCode//num_rot].append(patches_original[i].reshape(num_rot, secondLayerShape,secondLayerShape))
    originalPartsList = np.array(originalPartsList)
    for i in range(numParts):
        print(len(partsRegion[i]))
   
   
    k = np.zeros(numParts * num_rot)
    for i in range(trainingDataNum):
        for j in extractedFeature[i]:
            if j!=-1:
                k[j]+=1
    numSecondLayerParts = 10 

    print(k) 
    secondLayerPermutation = np.empty((num_rot, num_rot * num_rot * numParts), dtype = np.int_)
    for a in range(num_rot):
        if a == 0:
            secondLayerPermutation[a] = np.arange(num_rot * num_rot * numParts)
        else:
            secondLayerPermutation[a] = np.roll(secondLayerPermutation[a - 1], num_rot * num_rot * numParts)

    from pnet.bernoulli import em
    superParts = [em(np.array(partsRegion[i]).reshape(len(partsRegion[i]),-1),numSecondLayerParts,permutation=secondLayerPermutation,verbose=True) for i in range(numParts)]     
    allVisParts = []
    for i in range(numParts):
        print(superParts[i][3].shape)
        comps = np.array(superParts[i][3])
        raw_originals = np.array(originalPartsList[i])
        print(raw_originals.shape, comps.shape)
        visParts = np.asarray([raw_originals[comps[:,0]==k,(comps[comps[:,0]==k][:,1]-1)%num_rot].mean(0) for k in range(numParts)])
        allVisParts.append(visParts)

    gr.images(np.array(originalPartsList[0])[:20,0],show=False,zero_to_one=False, vmin=0,vmax=1,fileName = 'test.png')
    print(superParts[0][3][:20,:]) 


    print(shiftRotationLayer._visparts.shape)
    """
    Visualize the SuperParts
    """
    settings = {'interpolation':'nearest','cmap':plot.cm.gray,}
    settings['vmin'] = 0
    settings['vmax'] = 1
    plotData = np.ones(((2 + secondLayerShape)*100+2,(2+secondLayerShape)*(numSecondLayerParts + 1)+2))*0.8
    visualShiftParts = 0
    if 0:
        allPartsPlot = np.zeros((20,numSecondLayerParts + 1,12,12))
        gr.images(partsPlot.reshape(numParts,6,6),zero_to_one=False,vmin = 0, vmax = 1)
        allPartsPlot[:,0] = 0.5
        allPartsPlot[:,0,3:9,3:9] = partsPlot[20:40]
        allPartsPlot[:,1:,:,:] = allPartsLayerImg[20:40]
        gr.images(allPartsPlot.reshape(20 * (numSecondLayerParts + 1),12,12),zero_to_one=False, vmin = 0, vmax =1)
    elif 1:
        for i in range(numSecondLayerParts + 1):
            for j in range(numParts):
                if 0:#i == 0:
                    plotData[5 + j * (2 + secondLayerShape):5+firstLayerShape + j * (2 + secondLayerShape), 5 + i * (2 + secondLayerShape): 5+firstLayerShape + i * (2 + secondLayerShape)] = shiftRotationLayer._visparts[j+visualShiftParts]
                else:
                    plotData[2 + j * (2 + secondLayerShape):2 + secondLayerShape+ j * (2 + secondLayerShape),2 + i * (2 + secondLayerShape): 2+ secondLayerShape + i * (2 + secondLayerShape)] = allVisParts[j+visualShiftParts][i-1]
        plot.figure(figsize=(10,40))
        plot.axis('off')
        plot.imshow(plotData, **settings)
        plot.savefig('originalExParts1.pdf',format='pdf',dpi=900)
    else:
        pass


