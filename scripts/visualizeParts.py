import numpy as np
import amitgroup as ag
import pnet
import matplotlib.pyplot as plt

def extract(ims,allLayers):
    print(allLayers)
    curX = ims
    for layer in allLayers:
        curX = layer.extract(curX)
    return curX

# load the trained Image
X =  np.load('randomization.npy')
model = X.item()
# get the parts Layer
numParts1 = model['layers'][1]['num_parts']
net = pnet.PartsNet.load_from_dict(model)
allLayer = net.layers
print(allLayer)
ims,labels = ag.io.load_mnist('training') 
extractedFeature = []
for i in range(5):
    extractedFeature.append(extract(ims[0:1000],allLayer[0:i])[0])
extractedParts1 = extractedFeature[2]
partsPlot1 = np.zeros((numParts1,5,5))
partsCodedNumber1 = np.zeros(numParts1)

for i in range(1000):
    codeParts1 = extractedParts1[i].reshape(extractedParts1[i].shape[0:2])
    for m in range(24):
        for n in range(24):
            if(codeParts1[m,n]!=-1):
                partsPlot1[codeParts1[m,n]]+=ims[i,m:m+5,n:n+5] 
                partsCodedNumber1[codeParts1[m,n]]+=1
for j in range(numParts1):
    partsPlot1[j] = partsPlot1[j]/partsCodedNumber1[j]
partsCodedNumber1 = np.clip(partsCodedNumber1,0,2000)
plt.hist(partsCodedNumber1,bins = 50,color = 'blue')  
ag.plot.images(partsPlot1)
print(np.sum(partsCodedNumber1))
