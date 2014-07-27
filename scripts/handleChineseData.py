from amitgroup.io import load_casia
import argparse
import numpy as np
a = load_casia("training")
batchSize = 10000
totalBatch = int(np.floor(len(a[0])/batchSize))
print(len(a[0]))
print(totalBatch)


for k in range(totalBatch):
    b = a[0][k * batchSize:(k + 1) * batchSize]
    z = []

    for c in b:
        hSize = c.shape[0]
        wSize = c.shape[1]
        import numpy as np
        padSize = np.maximum(hSize,wSize)
        tempData = np.ones((padSize,padSize)) * 255
        halfPad = int((np.maximum(hSize,wSize) - np.minimum(hSize,wSize))/2)
        if(hSize > wSize):
            tempData[:,halfPad:halfPad + wSize] = c
        else:
            tempData[halfPad:halfPad + hSize,:] = c
        ratio = np.float(40/padSize)
        from scipy.ndimage import zoom
        c = zoom(tempData,ratio)
        z.append(c)
    z = np.asarray(z,dtype = np.float)
    z = z/255
    if 0:        
        from tables import *
        f = openFile('/var/tmp/.test/Chinese%d.h5' %k,'w')
        atom = Atom.from_dtype(z.dtype)
        ds = f.createCArray(f.root,'test',atom,z.shape)    
        ds[:] = z
        f.close()
    if 1:
        np.save('/var/tmp/.Chinese/Chinese%d.npy' %k,z)
        #print(z.shape)
        #z = (z > 0.8)
        
        #import amitgroup.plot as gr
        #gr.images(z,show=False,fileName = 'chinese.png')
