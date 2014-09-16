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
import sklearn.cluster

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

#     #############
#     # LOAD DATA #
#     #############

#     # Download the MNIST dataset if it is not present
#     data_dir, data_file = os.path.split(dataset)
#     if data_dir == "" and not os.path.isfile(dataset):
#         # Check if dataset is in the data directory.
#         new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
#         if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
#             dataset = new_path

#     if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
    import urllib
    #origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    #print ('Downloading data from %s' % origin)
    #urllib.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
#     f = gzip.open(dataset, 'rb')
#     train_set, valid_set, test_set = cPickle.load(f)
#     f.close()
    train_set = (shuffledExtract[:4000],shuffledLabel[:4000])
    valid_set = (shuffledExtract[4000:],shuffledLabel[4000:])
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        print(data_x.shape,data_y.shape)
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

#     test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]#,
#             (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=200):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
#     test_set_x, test_set_y = datasets[2]
    print(train_set_x.get_value(borrow=True).shape,train_set_y.shape,
          valid_set_x.get_value(borrow=True).shape,valid_set_y.shape,
#           test_set_x.get_value(borrow=True).shape,test_set_y.shape
    )
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    
#     n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=dataSize, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
#     test_model = theano.function(inputs=[index],
#             outputs=classifier.errors(y),
#             givens={
#                 x: test_set_x[index * batch_size: (index + 1) * batch_size],
#                 y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                print(validation_losses)
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

#                     test_losses = [test_model(i)
#                                    for i in xrange(n_test_batches)]
#                     test_score = numpy.mean(test_losses)

#                     print(('     epoch %i, minibatch %i/%i, test error of best'
#                        ' model %f %%') %
#                         (epoch, minibatch_index + 1, n_train_batches,
#                          test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    test_score = 0
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print ('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    #print >> sys.stderr, ('The code for file ' +
     #                     os.path.split(__file__)[1] +
      #                    ' ran for %.1fs' % ((end_time - start_time)))
    return classifier



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
    yhat = net.classify(ims)
    return yhat == labels

def testInvestigation(ims, labels, net):
    yhat = net.classify((ims,500))
    return np.where(yhat!=labels), yhat

    #X = np.load("testMay151.npy")
    #X = np.load("_3_100*6*6_1000*1*1_Jun_16_danny.npy")
    X = np.load("original6*6 2.npy")
    #X = np.load("sequential6*6.npy")
    model = X.item()
    # get num of Parts
    numParts = model['layers'][1]['num_parts']
    net = pnet.PartsNet.load_from_dict(model)
    allLayer = net.layers
    ims,labels = ag.io.load_mnist('training')
    trainingDataNum = 1000
    firstLayerShape = 6
    extractedFeature = extract(ims[0:trainingDataNum],allLayer[0:2])[0]
    print(extractedFeature.shape)
    extractedFeature = extractedFeature.reshape(extractedFeature.shape[0:3])
    partsPlot = np.zeros((numParts,firstLayerShape,firstLayerShape))
    partsCodedNumber = np.zeros(numParts)
    
    imgRegion= [[] for x in range(numParts)]
    partsRegion = [[] for x in range(numParts)]

    for i in range(trainingDataNum):
        codeParts = extractedFeature[i]
        for m in range(29 - firstLayerShape):
            for n in range(29 - firstLayerShape):
                if(codeParts[m,n]!=-1):
                    partsPlot[codeParts[m,n]]+=ims[i,m:m+firstLayerShape,n:n+firstLayerShape]
                    partsCodedNumber[codeParts[m,n]]+=1
    for j in range(numParts):
        partsPlot[j] = partsPlot[j]/partsCodedNumber[j]


    secondLayerCodedNumber = 0
    secondLayerShape = 12
    frame = (secondLayerShape - firstLayerShape)/2
    frame = int(frame)
    totalRange = 29 - firstLayerShape
    if 1:
        for i in range(trainingDataNum):
            codeParts = extractedFeature[i]
            for m in range(totalRange)[frame:totalRange - frame]:
                for n in range(totalRange)[frame:totalRange - frame]:
                    if(codeParts[m,n]!=-1):
                        imgRegion[codeParts[m,n]].append(ims[i, m - frame:m + secondLayerShape - frame,n - frame:n + secondLayerShape - frame])
                        secondLayerCodedNumber+=1
                        partsGrid = partsPool(codeParts[m-frame:m+frame + 1,n-frame:n+frame + 1],numParts)
                        partsRegion[codeParts[m,n]].append(partsGrid)
    
    
    newPartsRegion = []
    for i in range(numParts):
        newPartsRegion.append(np.asarray(partsRegion[i],dtype = np.uint8))
    np.save('/var/tmp/partsRegionOriginalJun29.npy',newPartsRegion)
    np.save('/var/tmp/imgRegionOriginalJun29.npy',imgRegion)
    ##second-layer parts
    numSecondLayerParts = 10
    allPartsLayer = [[pnet.PartsLayer(numSecondLayerParts,(1,1),
                        settings=dict(outer_frame = 0, 
                        threshold = 5, 
                        sample_per_image = 1, 
                        max_samples=10000, 
                        min_prob = 0.005,
                        #min_llh = -40
                        ))] 
                        for i in range(numParts)]
    allPartsLayerImg = np.zeros((numParts,numSecondLayerParts,secondLayerShape,secondLayerShape))
    allPartsLayerImgNumber = np.zeros((numParts,numSecondLayerParts))
    zeroParts = 0
    
    imgRegionPool = [[] for i in range(numParts * numSecondLayerParts)]
    for i in range(numParts):
        if(not partsRegion[i]):
            continue
        allPartsLayer[i][0].train_from_samples(np.array(partsRegion[i]),None)
        extractedFeaturePart = extract(np.array(partsRegion[i],dtype = np.uint8),allPartsLayer[i])[0]
        print(extractedFeaturePart.shape)
        for j in range(len(partsRegion[i])):
            if(extractedFeaturePart[j,0,0,0]!=-1):
                partIndex = extractedFeaturePart[j,0,0,0]
                allPartsLayerImg[i,partIndex]+=imgRegion[i][j]
                imgRegionPool[i * numSecondLayerParts + partIndex].append(imgRegion[i][j])
                allPartsLayerImgNumber[i,partIndex]+=1
            else:
                zeroParts+=1
    for i in range(numParts):
        for j in range(numSecondLayerParts):
            if(allPartsLayerImgNumber[i,j]):
                allPartsLayerImg[i,j] = allPartsLayerImg[i,j]/allPartsLayerImgNumber[i,j]
    
    
    if 1:
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
                    if i == 0:
                        plotData[5 + j * (2 + secondLayerShape):5+firstLayerShape + j * (2 + secondLayerShape), 5 + i * (2 + secondLayerShape): 5+firstLayerShape + i * (2 + secondLayerShape)] = partsPlot[j+visualShiftParts]
                    else:
                        plotData[2 + j * (2 + secondLayerShape):2 + secondLayerShape+ j * (2 + secondLayerShape),2 + i * (2 + secondLayerShape): 2+ secondLayerShape + i * (2 + secondLayerShape)] = allPartsLayerImg[j+visualShiftParts,i-1]
            plot.figure(figsize=(10,40))
            plot.axis('off')
            plot.imshow(plotData, **settings)
        else:
            pass


    
    digits = range(10)
    sup_ims = []
    sup_labels = []
    
    classificationTrainingNum = 1000
    for d in digits:
        ims0 = ag.io.load_mnist('training', [d], selection = slice(classificationTrainingNum), return_labels = False)
        sup_ims.append(ims0)
        sup_labels.append(d * np.ones(len(ims0),dtype = np.int64))
    sup_ims = np.concatenate(sup_ims, axis = 0)
    sup_labels = np.concatenate(sup_labels,axis = 0)

    #thirLevelCurx = np.load('./thirdLevelCurx.npy')
    thirLevelCurx = np.load('./thirdLevelCurx_LargeMatch.npy')[:5000]

    poolHelper = pnet.PoolingLayer(shape = (4,4),strides = (4,4))
    thirLevelCurx = np.array(thirLevelCurx, dtype = np.int64)
    pooledExtract = poolHelper.extract((thirLevelCurx[:,:,:,np.newaxis],500))
    print(pooledExtract.sum(axis = 3))
    print(pooledExtract.shape)
    sup_labels = sup_labels[:5000]
    sup_ims = sup_ims[:5000]
    index = np.arange(5000)
    randomIndex = np.random.shuffle(index)
    pooledExtract = pooledExtract.reshape(5000,-1)
    shuffledExtract = pooledExtract[index]
    shuffledLabel = sup_labels[index]

    dataSize = shuffledExtract.shape[1]
    classifier = sgd_optimization_mnist()
    weights = classifier.W.get_value(borrow=True)
    bias = classifier.b.get_value(borrow=True)
    weights = weights.reshape(4,4,500,10)
   
 
    trainingImg_curX = np.load('./thirdLevelCurx_LargeMatch.npy')[:1000]
    trainingImg_curX = np.array(trainingImg_curX, dtype = np.int64)
    pooledTrain = poolHelper.extract((trainingImg_curX[:,:,:,np.newaxis],500))
    trainImg,trainLabels = ag.io.load_mnist('training')
    newPooledExtract = np.array(pooledTrain[:1000]).reshape(1000,4,4,500)
    
    for p in range(4):
        for q in range(4):
            location1 = newPooledExtract[:,p,q,:]
            data = weights[p,q,:500,:]
            X = np.array(data.reshape(500,10),dtype=np.double)
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
                        if len(poolingIndex[m])==0:
                            continue
                        else:
    #                         print(poolingIndex[m][0])
                            newPooledExtract[r,p,q,:][poolingIndex[m][0]] = 1



    testImg_curX = np.load('./thirdLevelCurx_Test.npy')
    testImg_curX = np.array(testImg_curX, dtype = np.int64)
    pooledTest = poolHelper.extract((testImg_curX[:,:,:,np.newaxis],500))
    testImg,testLabels = ag.io.load_mnist('testing')


    newPooledExtractTest = np.array(pooledTest[:10000]).reshape(10000,4,4,500)
    for p in range(4):
        for q in range(4):
            location1 = newPooledExtractTest[:,p,q,:]
            data = weights[p,q,:500,:]
            X = np.array(data.reshape(500,10),dtype=np.double)
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
                        if len(poolingIndex[m])==0:
                            continue
                        else:
    #                         print(poolingIndex[m][0])
                            newPooledExtractTest[r,p,q,:][poolingIndex[m][0]] = 1


    #Train a class Model#

    classificationLayers = [pnet.SVMClassificationLayer(C = 1.0)]
    classificationNet = pnet.PartsNet(classificationLayers)
    classificationNet.train(np.array(newPooledExtract,dtype = np.int64), trainLabels[:1000]))
    print("Training Success!")
    testImg_Input = np.array(newPooledExtractTest, dtype = np.int64)
    testImg_batches = np.array_split(testImg_Input, 200)
    testLabels_batches = np.array_split(testLabels, 200)

    args = [tup + (classificationNet,) for tup in zip(testImg_batches, testLabels_batches)]
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



