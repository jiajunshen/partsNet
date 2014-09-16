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

"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import cPickle
import gzip
import time
import PIL.Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
import sklearn.cluster

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=784, n_hidden=200, \
        W=None, hbias=None, vbias=None, numpy_rng=None,
        theano_rng=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                      dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 6th output
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                    dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
                T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                      axis=1))

        return cross_entropy


def test_rbm(learning_rate=0.05, training_epochs=30,
             dataset='/Users/jiajunshen/Documents/Research/partsNet/data/mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=20):
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
    datasets = load_data(shuffledExtract,shuffledLabel)

    train_set_x, train_set_y = datasets[0]
#     test_set_x, test_set_y = datasets[2]
    numVisible = shuffledExtract.shape[1]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
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
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index], cost,
           updates=updates,
           givens={x: train_set_x[index * batch_size:
                                  (index + 1) * batch_size]},
           name='train_rbm')

    plotting_time = 0.
    start_time = time.clock()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
#         image = PIL.Image.fromarray(tile_raster_images(
#                  X=rbm.W.get_value(borrow=True).T,
#                  img_shape=(28, 28), tile_shape=(10, 10),
#                  tile_spacing=(1, 1)))
#         image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    #################################
    #     Sampling from the RBM     #
    #################################
#     # find out the number of test samples
#     number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

#     # pick random test examples, with which to initialize the persistent chain
#     test_idx = rng.randint(number_of_test_samples - n_chains)
#     persistent_vis_chain = theano.shared(numpy.asarray(
#             test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
#             dtype=theano.config.floatX))

#     plot_every = 1000
#     # define one step of Gibbs sampling (mf = mean-field) define a
#     # function that does `plot_every` steps before returning the
#     # sample for plotting
#     [presig_hids, hid_mfs, hid_samples, presig_vis,
#      vis_mfs, vis_samples], updates =  \
#                         theano.scan(rbm.gibbs_vhv,
#                                 outputs_info=[None,  None, None, None,
#                                               None, persistent_vis_chain],
#                                 n_steps=plot_every)

#     # add to updates the shared variable that takes care of our persistent
#     # chain :.
#     updates.update({persistent_vis_chain: vis_samples[-1]})
#     # construct the function that implements our persistent chain.
#     # we generate the "mean field" activations for plotting and the actual
#     # samples for reinitializing the state of our persistent chain
#     sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
#                                 updates=updates,
#                                 name='sample_fn')

#     # create a space to store the image for plotting ( we need to leave
#     # room for the tile_spacing as well)
#     image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains - 1),
#                              dtype='uint8')
#     for idx in xrange(n_samples):
#         # generate `plot_every` intermediate samples that we discard,
#         # because successive samples in the chain are too correlated
#         vis_mf, vis_sample = sample_fn()
#         print(' ... plotting sample ', idx)
#         image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
#                 X=vis_mf,
#                 img_shape=(28, 28),
#                 tile_shape=(1, n_chains),
#                 tile_spacing=(1, 1))
#         # construct image

#     image = PIL.Image.fromarray(image_data)
#     image.save('samples.png')
#     os.chdir('../')
    return rbm





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

def load_data(allDataX,allDataLabel):
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
    train_set = (allDataX[:5000],allDataLabel[:5000])
#     valid_set = (allDataX[4000:],allDataLabel[4000:])
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
#     valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y)]#, (valid_set_x, valid_set_y)]#,
#             (test_set_x, test_set_y)]
    return rval

    
#def trainPOP():
if pnet.parallel.main(__name__):
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
    #np.save("exPartsOriginalJun29.npy",allPartsLayer)
    
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
            plot.savefig('originalExParts_2.pdf',format='pdf',dpi=900)
        else:
            pass

    """
    Train A Class-Model Layer
    """
    
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
    testImg_curX = np.load('./thirdLevelCurx_Test.npy')[:5000]
    testImg_curX = np.array(testImg_curX, dtype = np.int64)
    pooledTest = poolHelper.extract((testImg_curX[:,:,:,np.newaxis],500))
    print(pooledExtract.sum(axis = 3))
    print(pooledExtract.shape)
    sup_labels = sup_labels[:5000]
    sup_ims = sup_ims[:5000]
    index = np.arange(5000)
    randomIndex = np.random.shuffle(index)
    pooledExtract = pooledExtract.reshape(5000,-1)
    shuffledExtract = pooledExtract[index]
    shuffledLabel = sup_labels[index]
    testImg = sup_ims[index]

    

    datasets = load_data(shuffledExtract,shuffledLabel)

    train_set_x, train_set_y = datasets[0]


    #testRbm = test_rbm()
    #weights = testRbm.W.get_value(borrow=True)
    #np.save('weights20Hidden.npy',weights)
    weights = np.load('weights20Hidden.npy')
    weights = weights.reshape(4,4,500,20)

    newsup_labels = []
    classificationTrainingNum = 100
    for d in digits:
        newsup_labels.append(d * np.ones(100,dtype = np.int64))
    sup_labels = np.concatenate(newsup_labels,axis = 0) 
    trainingImg_curX_all =  np.load('./thirdLevelCurx_LargeMatch.npy')
    trainingImg_curX = trainingImg_curX_all[:1000]
    for d in digits:
        trainingImg_curX[d * 100: (d + 1)*100] = trainingImg_curX_all[d * 1000: d*1000+100]
    trainingImg_curX = np.array(trainingImg_curX, dtype = np.int64)
    pooledTrain = poolHelper.extract((trainingImg_curX[:,:,:,np.newaxis],500))
    trainLabels = sup_labels

    newPooledExtract = np.array(pooledTrain[:1000]).reshape(1000,4,4,500)

    if 1:
        for p in range(4):
            for q in range(4):
                location1 = newPooledExtract[:,p,q,:]
                data = weights[p,q,:500,:]
                X = np.array(data.reshape(500,20),dtype=np.double)
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
                                #pass
    if 0:
        for p in range(5):
            print(trainLabels[p])
            gr.images(trainImg[p])
            for m in range(4):
                for n in range(4):
                   gr.images(np.array([allPartsLayerImg[(k%500)//10,k - ((k%500)//10) * 10] for k in np.where(newPooledExtract[p,m,n,:]==1)[0]]))

    testImg_curX = np.load('./thirdLevelCurx_Test.npy')
    testImg_curX = np.array(testImg_curX, dtype = np.int64)
    pooledTest = poolHelper.extract((testImg_curX[:,:,:,np.newaxis],500))
    testingNum = 1000
    testImg,testLabels = ag.io.load_mnist('testing')
    print(pooledTest.shape)
    newPooledExtractTest = np.array(pooledTest[:testingNum]).reshape(testingNum,4,4,500)
    if 1:
        for p in range(4):
            for q in range(4):
                location1 = newPooledExtractTest[:,p,q,:]
                data = weights[p,q,:500,:]
                X = np.array(data.reshape(500,20),dtype=np.double)
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

                for r in range(testingNum):
                    print(r)
                    for m in range(500):
                        if newPooledExtractTest[r,p,q,m] == 1:
                            if len(poolingIndex[m])==0:
                                continue
                            else:
        #                         print(poolingIndex[m][0])
                                newPooledExtractTest[r,p,q,:][poolingIndex[m][0]] = 1
                                #pass
    newPooledExtract = newPooledExtract.reshape(1000,-1)
    newPooledExtractTest = newPooledExtractTest.reshape(testingNum,-1)
    #Train a class Model#
    testLabels = testLabels[:testingNum]
    svmLayer = pnet.SVMClassificationLayer(C = 1.0)
    svmLayer.train(newPooledExtract[:1000], trainLabels[:1000])
    print("Training Success!")
    testImg_Input = np.array(newPooledExtractTest, dtype = np.int64)
    testImg_batches = np.array_split(newPooledExtractTest[:testingNum], 200)
    print(np.mean(svmLayer.extract(testImg_Input) == testLabels))
    if 0:
        testLabels_batches = np.array_split(testLabels, 200)

        args = [tup + (svmLayer,) for tup in zip(testImg_batches, testLabels_batches)]
        corrects = 0
        total = 0
        def format_error_rate(pr):
            return "{:.2f}%".format(100 * (1-pr))
        def clustering(X,L,layer):
            return L == layer.extract(X)
            
        print("Testing Starting...")

        for i, res in enumerate(pnet.parallel.starmap_unordered(clustering,args)):
            if i !=0 and i % 20 ==0:
                print("{0:05}/{1:05} Error rate: {2}".format(total, len(ims),format_error_rate(pr)))

            corrects += res.sum()
            print(res.sum())
            total += res.size

            pr = corrects / total

        print("Final error rate:", format_error_rate(pr))
