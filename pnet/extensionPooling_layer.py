from __future__ import division, print_function, absolute_import
from pnet.bernoullimm import BernoulliMM
from pnet.layer import Layer
import numpy as np
import numpy
import matplotlib.pyplot as plt
import theano
import cPickle
import gzip
import time
import PIL.Image
from pnet.rbm import RBM
import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams
@Layer.register('extension-parts-grouping-layer')
class ExtensionPoolingLayer(Layer):
    def __init__(self, n_parts, grouping_type = 'rbm', pooling_type = 'distance', pooling_distance = 5, weights_file = None, save_weights_file = None, settings = {}):
        self._n_parts = n_parts
        # The grouping_type can be rbm/mixture model
        self._grouping_type = grouping_type
        
        # The pooling_type can be distance/probability
        self._pooling_type = pooling_type

        # If the pooling_type is distance, then this parameter set the pooling distance
        self._pooling_distance = pooling_distance
        self._weights_file = weights_file
        self._save_weights_file = save_weights_file
        self._settings = settings
        self._pooling_matrix = None

    @property
    def trained(self):
        print("================================")
        return self._pooling_matrix is not None

    def train(self, X, Y = None, OriginalX = None):
        if self._weights_file is not None and self._grouping_type!= 'KLDistance':
            weights = np.load(self._weights_file)
            self._getPoolMatrix(weights,X.shape[1:], X)
        elif self._grouping_type == 'rbm':
            print("running RBM")
            
            n_hidden = self._settings.get('n_hidden', 200)
            training_epochs = self._settings.get('epochs', 50)
            learning_rate = self._settings.get('rate', 0.1)
            seed = self._settings.get('rbm_seed',123)
            rbmModel = testRBM(X.reshape(X.shape[0],-1), learning_rate = learning_rate, training_epochs = training_epochs, n_hidden = n_hidden, randomSeed = seed)

            weights = rbmModel.W.get_value(borrow=True)
            if self._save_weights_file is not None:
                np.save(self._save_weights_file,weights)
            self._getPoolMatrix(weights,X.shape[1:], X)
        elif self._grouping_type == 'mixture_model':
            #pass
            mixtureModel = BernoulliMM(n_components = 1000, n_iter = 100, n_init = 2, random_state = self._settings.get('em_seed', 0), min_prob = 0.005, joint=False,blocksize = 100)
            mixtureModel.fit(X.reshape((X.shape[0],-1)))
            print("mixtureModelweights")
            print(mixtureModel.weights_.shape)
            print(np.sum(mixtureModel.weights_))
            weights = mixtureModel.means_.reshape((mixtureModel.n_components,-1))
            #weights = np.log(weights/(1-weights))
            weights = np.swapaxes(weights,0,1)
            
            #joint_probability = mixtureModel.joint_probability
            component_weights = mixtureModel.weights_
            posterior = mixtureModel.posterior
            #print(np.sum(posterior,axis = 1))
            #print("posterior")
            #print(posterior.shape)
            #print(joint_probability.shape)
            if self._save_weights_file is not None:
                #np.save(self._save_weights_file,weights)
                np.save(self._save_weights_file,weights)
                np.save("./modelWeights.npy",mixtureModel.weights_)
                np.save("./modelMeans.npy",mixtureModel.means_)
                np.save("./modelPosterior.npy", posterior)
            #plt.hist(mixtureModel.weights_)
            #plt.show()
            #weights = posterior
            #self._getPoolMatrix(weights, X.shape[1:], X)
            self._getPoolMatrixByKLDistance(weights, X.shape[1:], X)
            #self._getPoolMatrixByMutual(weights, X.shape[1:],joint_probability,component_weights)
            #TODO: write mixture model weights generating.
        elif self._grouping_type == 'KLDistance':
            distanceFile = np.load(self._weights_file)
            self._getPoolMatrixByDistance(distanceFile, X.shape[1:], X)
        
    def _getPoolMatrixByKLDistance(self,weights_vector,data_shape, X):
        n_hiddenNodes = weights_vector.shape[1]
        assert self._n_parts == data_shape[2]
        weights_vector = weights_vector.reshape((data_shape[0],data_shape[1],self._n_parts, n_hiddenNodes))
        distanceMatrix = np.zeros((data_shape[0],data_shape[1],self._n_parts, self._n_parts))
        poolMatrix = np.zeros((data_shape[0],data_shape[1],self._n_parts, self._n_parts))
        partsCodedNum = np.sum(X, axis = 0)
        
        #TODO: Change the following part to cython
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for p in range(self._n_parts):
                    thisPoint = weights_vector[i,j,p,:]
                    for q in range(self._n_parts):
                        thatPoint = weights_vector[i,j,q,:]
                        distance = KLDistance(thisPoint, thatPoint) + KLDistance(thatPoint, thisPoint)
                        distanceMatrix[i,j,p,q] = distance
                    poolMatrix[i,j,p] = np.argsort(distanceMatrix[i,j,p,:])
                    for index in range(self._n_parts):
                        poolToPart = int(poolMatrix[i,j,p,index])
                        print("poolMatrix Num Investigation")
                        print(index, poolToPart,partsCodedNum[i,j,poolToPart])
                        if(partsCodedNum[i,j,poolToPart]<1):
                            poolMatrix[i,j,p,index] = p #Actually it makes more sense if we set it equal to -1. But in order to make [Reference: 01] easier, we set it equals to p here.    Reference Number : 02

        self._pooling_matrix = np.array(poolMatrix,dtype=np.int)
    
    def _getPoolMatrixByMutual(self,weights_vector,data_shape,joint_probability,component_weights):
        n_hiddenNodes = weights_vector.shape[1]
        assert self._n_parts == data_shape[2]
        weights_vector = weights_vector.reshape((data_shape[0],data_shape[1],self._n_parts, n_hiddenNodes))
        distanceMatrix = np.zeros((data_shape[0],data_shape[1],self._n_parts, self._n_parts))
        poolMatrix = np.zeros((data_shape[0],data_shape[1],self._n_parts, self._n_parts))
        joint_probability = joint_probability.reshape((n_hiddenNodes,3,data_shape[0],data_shape[1],self._n_parts,self._n_parts))
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for p in range(self._n_parts):
                    for q in range(self._n_parts):
                        mutualInformation = np.sum((np.log(joint_probability[:,:,i,j,p,q]) - np.log(weights_vector[i,j,p,:])[:,np.newaxis] - np.log(weights_vector[i,j,q,:])[:,np.newaxis]) * joint_probability[:,:,i,j,p,q],axis = 1)
                        distance = np.sum(component_weights * mutualInformation,axis = 0)
                        distanceMatrix[i,j,p,q] = distance
                    poolMatrix[i,j,p] = np.argsort(distanceMatrix[i,j,p,:])
        self._pooling_matrix = np.array(poolMatrix,dtype = np.int)
        #plt.hist(distanceMatrix[0,0,1,:])
        #plt.show()


    def _getPoolMatrix(self,weights_vector,data_shape, X):
        n_hiddenNodes = weights_vector.shape[1]
        assert self._n_parts == data_shape[2]
        weights_vector = weights_vector.reshape((data_shape[0],data_shape[1],self._n_parts, n_hiddenNodes))
        distanceMatrix = np.zeros((data_shape[0],data_shape[1],self._n_parts, self._n_parts))
        poolMatrix = np.zeros((data_shape[0],data_shape[1],self._n_parts, self._n_parts))
        partsCodedNum = np.sum(X, axis = 0)
        
        #TODO: Change the following part to cython
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                for p in range(self._n_parts):
                    thisPoint = weights_vector[i,j,p,:]
                    for q in range(self._n_parts):
                        thatPoint = weights_vector[i,j,q,:]
                        distance = np.sqrt(np.sum((thisPoint - thatPoint) * (thisPoint - thatPoint)))
                        distanceMatrix[i,j,p,q] = distance
                    poolMatrix[i,j,p] = np.argsort(distanceMatrix[i,j,p,:])
                    for index in range(self._n_parts):
                        poolToPart = int(poolMatrix[i,j,p,index])
                        #print("poolMatrix Num Investigation")
                        #print(index, poolToPart,partsCodedNum[i,j,poolToPart])
                        if(partsCodedNum[i,j,poolToPart]<1):
                            poolMatrix[i,j,p,index] = p #Actually it makes more sense if we set it equal to -1. But in order to make [Reference: 01] easier, we set it equals to p here.    Reference Number : 02

        self._pooling_matrix = np.array(poolMatrix,dtype=np.int)

    def _getPoolMatrixByDistance(self, distanceMatrix, data_shape, X):
        partsCodedNum = np.sum(X, axis = 0)

        poolMatrix = np.zeros((data_shape[0],data_shape[1],self._n_parts,self._n_parts))
        for p in range(self._n_parts):
            poolMatrix[:,:,p] = np.argsort(distanceMatrix[p,:])
            for index in range(self._n_parts):
                poolToPart = int(poolMatrix[0,0,p,index])
                if(np.sum(partsCodedNum[:,:,poolToPart])<1):
                    poolMatrix[:,:,p,index] = p # Refer to [Reference: 02]

        self._pooling_matrix = np.array(poolMatrix, dtype = np.int)

    def extract(self,X):
        assert self._pooling_matrix is not None, "Must be trained before calling extract"
        assert X.ndim == 4, "Input X dimension is not correct"
        assert X.shape[3] == self._n_parts, "the parts number is not correct"

        for k in range(X.shape[0]):
            for i in range(X.shape[1]):
                for j in range(X.shape[2]):
                    for m in range(X.shape[3]):
                        if X[k,i,j,m] == 1:
                            if self._pooling_type == 'distance':
                                X[k,i,j,:][self._pooling_matrix[i,j,m,:self._pooling_distance]] = 1#                              Reference Number: 01
                            #elif self._pooling_type == 'probability':
                            #    PooledLocation = poolByProbability(self._pooling_matrix[i,j,m,:se])
                            #    X[k,i,j,:]
        return X

    def save_to_dict(self):
        d = {}
        d['n_parts'] = self._n_parts
        d['grouping_type'] = self._grouping_type
        d['pooling_type'] = self._pooling_type
        d['pooling_distance'] = self._pooling_distance
        d['weights_file'] = self._weights_file
        d['save_weights_file'] = self._save_weights_file
        d['settings'] = self._settings
        d['pooling_matrix'] = self._pooling_matrix
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['n_parts'], d['grouping_type'], d['pooling_type'], d['pooling_distance'], d['weights_file'], d['save_weights_file'], d['settings'])
        obj._pooling_matrix = d['pooling_matrix']
        return obj





def load_data(allDataX):
    import urllib

    print('... loading data')

    train_set = (allDataX)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x = data_xy
        print(data_x.shape)
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x

    train_set_x = shared_dataset(train_set)

    rval = [train_set_x]
    return rval


def testRBM(datasets, learning_rate=0.1, training_epochs=50,
              batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=200, randomSeed=123):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    numVisible = datasets.shape[1]
    datasets = load_data(datasets)
    train_set_x = datasets[0]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(randomSeed)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible= numVisible,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index], cost,
           updates=updates,
           givens={x: train_set_x[index * batch_size:
                                  (index + 1) * batch_size]},
           name='train_rbm')

    plotting_time = 0.
    start_time = time.clock()
    print("training")
    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = time.clock()
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    return rbm


def KLDistance(p,q):
    p = np.ravel(p)
    q = np.ravel(q)
    return np.sum(p * np.log(p/q))
