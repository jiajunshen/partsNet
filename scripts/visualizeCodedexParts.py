from __future__ import division, print_function,absolute_import
import pylab as plt
import amitgroup.plot as gr
import numpy as np
import amitgroup as ag
import os
import pnet
import matplotlib.pylab as plot
from pnet.cyfuncs import index_map_pooling
from queue import Queue
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
    yhat = net.classify((ims,2000))
    return yhat == labels
    


#def trainPOP():
if pnet.parallel.main(__name__):
    #X = np.load("testMay151.npy")
    #X = np.load("_3_100*6*6_1000*1*1_Jun_16_danny.npy")
    #X = np.load("Jun26Rot.npy")
    X = np.load("Jun29Rot.npy")
    #X = np.load("sequential6*6.npy")
    model = X.item()
    # get num of Parts
    numParts = model['layers'][0]['num_true_parts'] * model['layers'][0]['num_orientations']
    net = pnet.PartsNet.load_from_dict(model)
    allLayer = net.layers
    ims,labels = ag.io.load_mnist('training')
    ORI = 8
    trainingDataNum = 1000 * ORI
    firstLayerShape = 6
    from skimage import transform
    allTrainData = []
    angles = np.arange(0,360,360/ORI)
    for i in range(1000):
        allTrainData.append(np.asarray([transform.rotate(ims[i],angle,resize=False,mode = 'nearest') for angle in angles]))
    allTrainData = np.asarray(allTrainData)
    print(allTrainData.shape)
    allTrainData = allTrainData.reshape((1000 * angles.shape[0],) +  allTrainData.shape[2:])
    print(allTrainData.shape)
    #gr.images(allTrainData[:200],show=False,fileName = 'rotataion.png')
    
    if 1:
        #extractedFeature = allLayer[0].extract(allTrainData)[0]
        #np.save('extractedFeatureJun29.npy',extractedFeature)
        extractedFeature = np.load('extractedFeatureJun29.npy')
        partsPlot = np.zeros((numParts,firstLayerShape,firstLayerShape))
        partsCodedNumber = np.zeros(numParts)
        
        imgRegion= [[] for x in range(numParts)]
        partsRegion = [[] for x in range(numParts)]
        print(extractedFeature.shape)
        print(trainingDataNum) 
        secondLayerCodedNumber = 0
        secondLayerShape = 12
        frame = (secondLayerShape - firstLayerShape)/2
        frame = int(frame)
        totalRange = 29 - firstLayerShape
        numSecondLayerParts = 20
        
        #allLayer = np.load('firstLayerInformationJun29.npy')
        allPartsLayer = np.load('exPartsJun29.npy')
        partsRegion = np.load('/var/tmp/partsRegionJun29.npy')

    allPartsLayerImgNumber = np.zeros((numParts,numSecondLayerParts))
    allPartsLayerMeanImg = np.zeros((numParts,numSecondLayerParts,secondLayerShape,secondLayerShape))
    allPartsLayerImg = [[[] for j in range(numSecondLayerParts)] for i in range(numParts)]
    imgRegion = np.load('/var/tmp/imgRegionJun29.npy')
    notTrain = 0;
    for i in range(numParts):
        if(allPartsLayer[i][0].trained == False):
            notTrain+=1
            continue
        extractedFeaturePart = extract(partsRegion[i],allPartsLayer[i])[0]
        for j in range(extractedFeaturePart.shape[0]):
            if(extractedFeaturePart[j,0,0,0]!=-1):
                partIndex = extractedFeaturePart[j,0,0,0]
                allPartsLayerImg[i][partIndex].append(allTrainData[imgRegion[i][j]])
                allPartsLayerMeanImg[i,partIndex,:]+=allTrainData[imgRegion[i][j]]
                allPartsLayerImgNumber[i,partIndex]+=1
    print(allPartsLayerImgNumber)
    newAllPartsLayerImg =  [[[] for j in range(numSecondLayerParts)] for i in range(numParts)]
    for i in range(numParts):
        for j in range(numSecondLayerParts):
            allPartsLayerMeanImg[i,j,:] = allPartsLayerMeanImg[i,j,:]/allPartsLayerImgNumber[i,j]
            for k in range(len(allPartsLayerImg[i][j])):
                newAllPartsLayerImg[i][partIndex].append(np.asarray(allPartsLayerImg[i][j][k]))
    import matplotlib.pylab as plot
    import random
    print("==================================================")
    print(notTrain)
    settings = {'interpolation':'nearest','cmap':plot.cm.gray,}
    settings['vmin'] = 0
    settings['vmax'] = 1
    plotData = np.ones(((2 + secondLayerShape) * 4 * 10, (2 + secondLayerShape) * (20 + 1) + 2)) * 0.8 
    a = [random.randint(0,numParts-1) for i in range(10)]
    for i in range(len(a)):
        b = [random.randint(0,numSecondLayerParts-1) for m in range(4)]
        for j in range(len(b)):
            c = [random.randint(0,len(allPartsLayerImg[a[i]][b[j]])-1) for n in range(np.minimum(20,len(allPartsLayerImg[a[i]][b[j]])-1 ) )]
            
            print(c)
            plotData[2 + (4 * i + j) * (2 + secondLayerShape):2 + (4 * i + j) * (2 + secondLayerShape) + 12, 2:2 + secondLayerShape] = allPartsLayerMeanImg[a[i],b[j]]
            for k in range(len(c)):
                print(allPartsLayerImg[a[i]][b[j]][c[k]].shape) 
                plotData[2 + (4 * i + j) * (2 + secondLayerShape):2 + (4 * i + j) * (2 + secondLayerShape) + 12, 2 + (k + 1) * (2 + secondLayerShape):2 + secondLayerShape + (k + 1) *(2 + secondLayerShape)] = allPartsLayerImg[a[i]][b[j]][c[k]]
    plot.figure(figsize=(10,40))
    plot.axis('off')
    plot.imshow(plotData,**settings)
    plot.savefig('visualizePatchesCodedtoExParts.pdf',format='pdf',dpi = 900)
     
            







