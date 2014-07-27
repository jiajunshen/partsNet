import numpy as np
import tables
def partsPool(originalPartsRegion,numParts):
    partsGrid = np.zeros((1,1,numParts))
    for i in range(originalPartsRegion.shape[0]):
        for j in range(originalPartsRegion.shape[1]):
            if(originalPartsRegion[i,j]!=-1):
                partsGrid[0,0,originalPartsRegion[i,j]] = 1
    return partsGrid
import pnet
if pnet.parallel.main(__name__):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rot',metavar='<index of rot>',type = int)
    args = parser.parse_args()
    rot = args.rot
    #extractedFeature = np.load('extractedFeature.npy')
    extractedFeature = np.load('extractedFeatureJun29.npy')
    partsRegion = [[] for x in range(50)]
    for i in range(8000):
        print(i)
        codeParts = extractedFeature[i]
        for m in range(23)[3:20]:
            for n in range(23)[3:20]:
                if(np.mod(codeParts[m,n],16) == rot):
                    partsGrid = partsPool(codeParts[m-3:m+4,n-3:n+4],400)
                    partsRegion[codeParts[m,n]//16].append(partsGrid)
    newPartsRegion = []
    for i in range(50):
        newPartsRegion.append(np.asarray(partsRegion[i],dtype = np.uint8))
    #f = tables.openFile('poolExtractedFeatureRot0.h5',mode = 'w')
    #root = f.root
    #atom = tables.Atom.from_dtype(partsRegion.dtype)
    #ds = f.createCArray(root,'array',atom,partsRegion.shape)
    #ds[:] = partsRegion
    #f.close()
    np.save('/var/tmp/poolExtractedFeatureJun29Rot%d.npy' %rot, newPartsRegion)
