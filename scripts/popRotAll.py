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
        for i in range(trainingDataNum):
            codeParts = extractedFeature[i].reshape(29 - firstLayerShape, 29 - firstLayerShape)
            for m in range(29 - firstLayerShape):
                for n in range(29 - firstLayerShape):
                    if(codeParts[m,n]!=-1):
                        partsPlot[codeParts[m,n]]+=allTrainData[i,m:m+firstLayerShape,n:n+firstLayerShape]
                        partsCodedNumber[codeParts[m,n]]+=1
        for j in range(numParts):
            partsPlot[j] = partsPlot[j]/partsCodedNumber[j]
        
        #gr.images(partsPlot[:200],show=False,fileName = 'firstLayerParts.png')

        secondLayerCodedNumber = 0
        secondLayerShape = 12
        frame = (secondLayerShape - firstLayerShape)/2
        frame = int(frame)
        totalRange = 29 - firstLayerShape
    
    if 0:
        largerRegionOrientedPartsLayer = pnet.OrientedPartsLayer(50, 16, (secondLayerShape,secondLayerShape),settings = dict(outer_frame = 3, em_seed = 0, n_init = 2, threshold = 2,samples_per_image = 40, max_samples = 10000, train_limit = 20000, rotation_spreading_radius = 0, min_prob= 0.0005, bedges = dict(k = 5, minimum_contrast = 0.05, spread = 'orthogonal', radius = 1, contrast_insensitive = False,)))
        largerRegionData = largerRegionOrientedPartsLayer._get_patches(ims)
        print(largerRegionData.shape)
    if 1:
        if 0:
            for i in range(trainingDataNum):
                print("code small regions", i)
                codeParts = extractedFeature[i]
                for m in range(totalRange)[frame:totalRange - frame]:
                    for n in range(totalRange)[frame:totalRange - frame]:
                        if(codeParts[m,n]!=-1):
                        #if(codeParts[m,n] <= 3):
                            imgRegion[codeParts[m,n]].append((i, slice(m - frame,m + secondLayerShape - frame),slice(n - frame,n + secondLayerShape - frame)))
                            #secondLayerCodedNumber+=1
                            partsGrid = partsPool(codeParts[m-frame:m+frame + 1,n-frame:n+frame + 1],numParts)
                            partsRegion[codeParts[m,n]].append(partsGrid)
        newPartsRegion = []
        for i in range(numParts):
            newPartsRegion.append(np.asarray(partsRegion[i],dtype = np.uint8))
        #np.save('/var/tmp/partsRegionJun29.npy',newPartsRegion)
        #np.save('/var/tmp/imgRegionJun29.npy',imgRegion)
        partsRegion = np.load('/var/tmp/partsRegionJun29.npy')
        imgRegion = np.load('/var/tmp/imgRegionJun29.npy')
        #np.save('partsRegion.npy',partsRegion) 
        #np.save('/var/tmp/imgRegion.npy',imgRegion)
        #RotNo = 1
        #imgRegion = np.load('/var/tmp/imgRegion.npy')
        ##second-layer parts
        #partsRegion = np.load('/var/tmp/partsRegionJun29.npy')
        #numTrueParts = 50
        numSecondLayerParts = 20

        if 0:
            allPartsLayer = [[pnet.PartsLayer(numSecondLayerParts,(1,1),
                                settings=dict(outer_frame = 0, 
                                threshold = 5, 
                                sample_per_image = 1, 
                                max_samples=10000, 
                                min_prob = 0.005))] 
                                for i in range(numParts)]
            allPartsLayerImg = np.zeros((numParts,numSecondLayerParts,secondLayerShape,secondLayerShape))
            allPartsLayerImgNumber = np.zeros((numParts,numSecondLayerParts))
            zeroParts = 0
            
            #imgRegionPool = [[] for i in range(numParts * numSecondLayerParts)]



            print("Start to train exParts")
            #for i in range(numParts):
            for i in range(numParts):
                if(partsRegion[i].shape[0] <= 30):
                    continue
                allPartsLayer[i][0].train_from_samples(partsRegion[i],None)
                extractedFeaturePart = extract(partsRegion[i],allPartsLayer[i])[0]
                print(extractedFeaturePart.shape)
                for j in range(extractedFeaturePart.shape[0]):
                    if(extractedFeaturePart[j,0,0,0]!=-1):
                        partIndex = extractedFeaturePart[j,0,0,0]
                        allPartsLayerImg[i,partIndex]+=allTrainData[imgRegion[i][j]]
                        #imgRegionPool[i * numSecondLayerParts + partIndex].append(imgRegion[i][j])
                        allPartsLayerImgNumber[i,partIndex]+=1
                    else:
                        zeroParts+=1

            print("handling visualization exParts")
            for i in range(numParts):
                for j in range(numSecondLayerParts):
                    if(allPartsLayerImgNumber[i,j]):
                        allPartsLayerImg[i,j] = allPartsLayerImg[i,j]/allPartsLayerImgNumber[i,j]


            print("visualize")
            
            """
            Visualize the SuperParts
            """
            settings = {'interpolation':'nearest','cmap':plot.cm.gray,}
            settings['vmin'] = 0
            settings['vmax'] = 1
            plotData = np.ones(((2 + secondLayerShape)*400+2,(2+secondLayerShape)*(numSecondLayerParts + 1)+2))*0.8
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
                        if i == 0:
                            plotData[5 + j * (2 + secondLayerShape):5+firstLayerShape + j * (2 + secondLayerShape), 5 + i * (2 + secondLayerShape): 5+firstLayerShape + i * (2 + secondLayerShape)] = partsPlot[j+visualShiftParts]
                        else:
                            plotData[2 + j * (2 + secondLayerShape):2 + secondLayerShape+ j * (2 + secondLayerShape),2 + i * (2 + secondLayerShape): 2+ secondLayerShape + i * (2 + secondLayerShape)] = allPartsLayerImg[j,i-1]
                plot.figure(figsize=(10,40))
                plot.axis('off')
                plot.imshow(plotData, **settings)
                plot.savefig('rotExParts.pdf',format='pdf',dpi=900)
            else:
                pass

        #np.save('firstLayerInformationJun29.npy',allLayer[0:2])
        allPartsLayer = np.load('exPartsJun29.npy')

        """
        Train A Class-Model Layer
        """
        
        digits = range(10)
        sup_ims = []
        sup_labels = []
        
        classificationTrainingNum = 100
        for d in digits:
            ims0 = ag.io.load_mnist('training', [d], selection = slice(classificationTrainingNum), return_labels = False)
            sup_ims.append(ims0)
            sup_labels.append(d * np.ones(len(ims0),dtype = np.int64))
        sup_ims = np.concatenate(sup_ims, axis = 0)
        sup_labels = np.concatenate(sup_labels,axis = 0)
        
        print(sup_ims.shape)
        curX = allLayer[0].extract(sup_ims)[0]
        print(curX.shape)
        curX = curX.reshape(curX.shape[0:3])
        print(curX.dtype)
        secondLevelCurx = np.zeros((10 * classificationTrainingNum,29 - secondLayerShape,29 - secondLayerShape,1,1,numParts))
        secondLevelCurxCenter = np.zeros((10 * classificationTrainingNum,29- secondLayerShape,29 - secondLayerShape))
        #for i in range(10 * classificationTrainingNum):
        #    codeParts = curX[i]
        for m in range(totalRange)[frame:totalRange-frame]:
            for n in range(totalRange)[frame:totalRange-frame]:
                secondLevelCurx[:,m-frame,n-frame] = index_map_pooling(np.asarray(curX[:,m-frame:m+frame+1,n-frame:n+frame+1],dtype = np.int64),numParts,(2 * frame + 1,2 * frame + 1),(2 * frame + 1,2 * frame + 1))
                secondLevelCurxCenter[:,m-frame,n-frame] = curX[:,m,n]

        thirdLevelCurx = np.zeros((10 * classificationTrainingNum, 29 - secondLayerShape,29 - secondLayerShape))
        for i in range(int(10 * classificationTrainingNum)):
            for m in range(29 - secondLayerShape):
                for n in range(29 - secondLayerShape):
                    if(secondLevelCurxCenter[i,m,n]!=-1):
                        firstLevelPartIndex = secondLevelCurxCenter[i,m,n]
                        #print(firstLevelPartIndex)
                        firstLevelPartIndex = int(firstLevelPartIndex)
                        extractedFeaturePart = extract(np.array(secondLevelCurx[i,m,n][np.newaxis,:],dtype = np.uint8),allPartsLayer[firstLevelPartIndex])[0]
                        #print("secondLayerExtraction")
                        #print(extractedFeaturePart.shape)
                        thirdLevelCurx[i,m,n] = int(numSecondLayerParts * firstLevelPartIndex + extractedFeaturePart)
                        #print(numSecondLayerParts,firstLevelPartIndex,extractedFeaturePart,thirdLevelCurx[i,m,n])
                    else:
                        thirdLevelCurx[i,m,n] = -1
        
        print(thirdLevelCurx.shape)
        #return thirdLevelCurx,allPartsLayerImg 
        if 1:
            classificationLayers = [
                                pnet.PoolingLayer(shape = (4,4),strides = (4,4)),
                                #pnet.MixtureClassificationLayer(n_components = 5, min_prob = 1e-7, block_size = 20)
                                pnet.SVMClassificationLayer(C=1.0)
            ]
            classificationNet = pnet.PartsNet(classificationLayers)
            classificationNet.train((np.array(thirdLevelCurx[:,:,:,np.newaxis],dtype = np.int64),int(numParts * numSecondLayerParts)),sup_labels[:])
            print("Training Success!!")    
        
        if 1:    
            testImg,testLabels = ag.io.load_mnist('testing')
            testingNum = testLabels.shape[0]
            print("training extract Begin") 
            curTestX = extract(testImg, allLayer[0:2])[0]
            print("training extract End")
            curTestX = curTestX.reshape(curTestX.shape[0:3])
            secondLevelCurTestX = np.zeros((testingNum, 29 - secondLayerShape,29 - secondLayerShape,1,1,numParts))
            secondLevelCurTestXCenter = np.zeros((testingNum, 29 - secondLayerShape,29 - secondLayerShape))
            
            import time
            start = time.time()
            #for i in range(testingNum):
            #    codeParts = curTestX[i]
            for m in range(totalRange)[frame:totalRange - frame]:
                for n in range(totalRange)[frame:totalRange-frame]:
                    secondLevelCurTestX[:,m-frame,n-frame] = index_map_pooling(curTestX[:,m-frame:m+frame + 1,n-frame:n+frame + 1],numParts,(2 * frame + 1,2 * frame + 1),(2 * frame + 1,2 * frame + 1))
                    secondLevelCurTestXCenter[:,m-frame,n-frame] = curTestX[:,m,n]
            afterPool = time.time()
            print(afterPool - start)
            thirdLevelCurTestX = np.zeros((testingNum, 29 - secondLayerShape, 29 - secondLayerShape))
            featureMap = [[] for i in range(numParts)]
            for i in range(testingNum):
                for m in range(29 - secondLayerShape):
                    for n in range(29 - secondLayerShape):
                        if(secondLevelCurTestXCenter[i,m,n]!=-1):
                            firstLevelPartIndex = int(secondLevelCurTestXCenter[i,m,n])
                            featureMap[firstLevelPartIndex].append(np.array(secondLevelCurTestX[i,m,n],dtype = np.uint8))
                            #extractedFeaturePart = extract(np.array(secondLevelCurTestX[i,m,n][np.newaxis,:],dtype = np.uint8),allPartsLayer[firstLevelPartIndex])[0]
                            #thirdLevelCurTestX[i,m,n] = numSecondLayerParts * firstLevelPartIndex + extractedFeaturePart
                        #else:
                            #thirdLevelCurTestX[i,m,n] = -1
            extractedFeatureMap = [Queue() for i in range(numParts)]
            for i in range(numParts):
                partFeatureMap = np.array(featureMap[i],dtype = np.uint8)
                allExtractedFeature = extract(np.array(partFeatureMap),allPartsLayer[i])[0]
                for feature in allExtractedFeature:
                    extractedFeatureMap[i].put(feature)
            
            for i in range(testingNum):
                for m in range(29 - secondLayerShape):
                    for n in range(29 - secondLayerShape):
                        if(secondLevelCurTestXCenter[i,m,n]!=-1):
                            firstLevelPartIndex = int(secondLevelCurTestXCenter[i,m,n])
                            if(extractedFeatureMap[firstLevelPartIndex].qsize()==0):
                                print("something is wrong")
                                extractedFeaturePart = -1
                            else:
                                extractedFeaturePart = extractedFeatureMap[firstLevelPartIndex].get()
                            thirdLevelCurTestX[i,m,n] = numSecondLayerParts * firstLevelPartIndex + extractedFeaturePart
                        else:
                            thirdLevelCurTestX[i,m,n] = -1
            end = time.time()
            print(end-afterPool)
            print(thirdLevelCurTestX.shape)
            testImg_Input = np.array(thirdLevelCurTestX[:,:,:,np.newaxis],dtype = np.int64) 
            testImg_batches = np.array_split(testImg_Input,200)
            testLabels_batches = np.array_split(testLabels, 200)
            
            args = [tup + (classificationNet,) for tup in zip(testImg_batches,testLabels_batches)]
            
            corrects = 0
            total = 0
            
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













