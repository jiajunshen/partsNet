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
    #X = np.load("Jun29Rot.npy")
    #X = np.load("sequential6*6.npy")
    X = np.load("original6*6.npy")
    model = X.item()
    # get num of Parts
    #numParts = model['layers'][0]['num_true_parts'] * model['layers'][0]['num_orientations']
    numParts = model['layers'][1]['num_parts']
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
        #extractedFeature = np.load('extractedFeatureJun29.npy')
        partsPlot = np.zeros((numParts,firstLayerShape,firstLayerShape))
        partsCodedNumber = np.zeros(numParts)
        
        imgRegion= [[] for x in range(numParts)]
        partsRegion = [[] for x in range(numParts)]
        #print(extractedFeature.shape)
        print(trainingDataNum) 
        secondLayerCodedNumber = 0
        secondLayerShape = 12
        frame = (secondLayerShape - firstLayerShape)/2
        frame = int(frame)
        totalRange = 29 - firstLayerShape
        numSecondLayerParts = 20
        
        #allLayer = np.load('firstLayerInformationJun29.npy')
        allPartsLayer = np.load('exPartsJun29.npy')
        #partsRegion = np.load('/var/tmp/partsRegionJun29.npy')
        partsRegion = np.load('/var/tmp/partsRegionOriginalJun29.npy')
    allPartsLayerImgNumber = np.zeros((numParts,numSecondLayerParts))
    allPartsLayerMeanImg = np.zeros((numParts,numSecondLayerParts,secondLayerShape,secondLayerShape))
    allPartsLayerImg = [[[] for j in range(numSecondLayerParts)] for i in range(numParts)]
    #imgRegion = np.load('/var/tmp/imgRegionJun29.npy')
    imgRegion = np.load('/var/tmp/imgRegionOriginalJun29.npy')
    notTrain = 0;
    for i in range(numParts):
        if(allPartsLayer[i][0].trained == False):
            notTrain+=1
            continue
        extractedFeaturePart = extract(partsRegion[i],allPartsLayer[i])[0]
        for j in range(extractedFeaturePart.shape[0]):
            if(extractedFeaturePart[j,0,0,0]!=-1):
                partIndex = extractedFeaturePart[j,0,0,0]
                #allPartsLayerImg[i][partIndex].append(allTrainData[imgRegion[i][j]])
                #allPartsLayerMeanImg[i,partIndex,:]+=allTrainData[imgRegion[i][j]]
                
                allPartsLayerImg[i][partIndex].append(imgRegion[i][j])
                allPartsLayerMeanImg[i,partIndex,:]+=imgRegion[i][j]
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

    secondLevelCurTestX = np.zeros((25, 29 - secondLayerShape, 29 - secondLayerShape,1,1,numParts))
    secondLevelCurTestXCenter = np.zeros((25, 29 - secondLayerShape, 29 - secondLayerShape))
    codeData = ims[0:25]
    curTest = extract(codeData,allLayer[0:2])[0]#allLayer[0].extract(codeData)[0]
    curTest = curTest.reshape(curTest.shape[0:3])
    print(curTest.shape) 
    for m in range(totalRange)[frame:totalRange - frame]:
        for n in range(totalRange)[frame:totalRange - frame]:
            print(m,n)
            secondLevelCurTestX[:,m-frame,n-frame] = index_map_pooling(curTest[:,m-frame:m+frame+1,n-frame:n+frame+1],numParts,(2 * frame + 1,2*frame+1), (2*frame+1, 2* frame+1))
            secondLevelCurTestXCenter[:,m-frame, n-frame] = curTest[:,m,n]
    thirdLevelCurTestX = np.zeros((25,29 - secondLayerShape,29 - secondLayerShape))

    plotData = np.ones(((10 + (2 + secondLayerShape) * (29 - secondLayerShape)) * 5 + 10,(10 + (2 + secondLayerShape) * (29 - secondLayerShape)) * 5 + 10)) * 0.8
    for p in range(5):
        for q in range(5):
            i = 5 * p + q
            for m in range(29 - secondLayerShape):
                for n in range(29 - secondLayerShape):
                    if(secondLevelCurTestXCenter[i,m,n]!=-1):
                        firstLevelPartIndex = int(secondLevelCurTestXCenter[i,m,n])
                        extractedFeaturePart = extract(np.array(secondLevelCurTestX[i,m,n][np.newaxis,:],dtype = np.uint8),allPartsLayer[firstLevelPartIndex])[0]
                        plotData[ (10 + (2 + secondLayerShape) * (29 - secondLayerShape)) * p + (2 + secondLayerShape) * m: (10 + (2 + secondLayerShape) * (29 - secondLayerShape)) * p + (2 + secondLayerShape) * m + 12, (10 + (2 + secondLayerShape) * (29 - secondLayerShape)) * q + (2 + secondLayerShape) * n:(10 + (2 + secondLayerShape) * (29 - secondLayerShape)) * q + (2 + secondLayerShape) * n + 12] = allPartsLayerMeanImg[firstLevelPartIndex,extractedFeaturePart]
    

    plot.figure(figsize=(10,40))
    plot.axis('off')
    plot.imshow(plotData,**settings)
    plot.savefig('visualizePatchesCodedtoExParts.pdf',format='pdf',dpi = 900)
     
            







