__author__ = 'jiajunshen'
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
from multiprocessing import Pool, Value, Array
import theano
import theano.tensor as T
#from theano import RandomStreams


class multiLogisticRegression(object):
    """
    Number of models : K
    Number of training data for each models: N
    Dimension of each training data: F
    Dimension of total data: K x N x F
    Dimension of data label: K x N
    dimension of the weights: K x F
    dimension of the bias: K
    Because this is for multi-label logistic regression, the input dimension would be
    """
    def __init__(
        self,
        input_data = None,
        input_label = None,
        n_feature = None,
        n_model = None,
        W = None,
        bias = None,
        numpy_rng = None,
        theano_rng = None,
        sample_per_model = None,
        trainingSteps = 1000
    ):
        if input_data != None:
            assert input_data.shape[0] == n_model
            assert input_data.shape[1] == sample_per_model
            assert input_data.shape[2] == n_feature
        if input_label != None:
            assert input_label.shape[0] == n_model
            assert input_label.shape[1] == sample_per_model
        """
        multi-label logisticRegression constructor
        """
        self.n_feature = n_feature
        self.n_model = n_model
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        #if theano_rng is None:
        #    theano_rng = RandomStreams(numpy_rng.randint(2**30))


        if W is None:
            init_w = np.asarray(numpy_rng.uniform(low = -1, high = 1, size=(n_model, n_feature)), dtype=theano.config.floatX)
            W = init_w

        if bias is None:
            bias = np.zeros((n_model, 1), dtype = theano.config.floatX)

        self.input_data = input_data
        #if not input_data:
        #    self.input_data = T.tensor3('x',dtype=theano.config.floatX)
        self.label = input_label
        #if not input_label:
        #    self.input_label = T.matrix('y')

        self.W = W
        self.bias = bias
        #self.theano_rng = theano_rng
        self.training_steps = trainingSteps
        self.sample_per_model = sample_per_model

    def train(self, input_data = None, input_label = None):
        if input_data is None or input_label is None:
            assert self.input_data is not None and self.input_label is not None
        else:
            self.input_data = input_data.astype(theano.config.floatX)
            self.input_label = input_label.astype(theano.config.floatX)

        x = T.tensor3('x', dtype=theano.config.floatX)
        y = T.matrix('y')
        w = theano.shared((self.W).astype(theano.config.floatX), name = "W")
        b = theano.shared((self.bias).astype(theano.config.floatX), name = "b", broadcastable = [False, True])

        p_1 = 1 / (1 + T.exp(- T.batched_dot(x, w) - b)) # probability of having a one shape : k * n
        prediction = p_1 > 0.5 # The prediction that is done: 0 or 1  shape: k * n
        xent = - y * T.log(p_1) - (1 - y) * T.log(1 - p_1) # shape k * n
        cost = xent.mean(axis = 1).sum() + 0.01 * (w ** 2).sum() # scalar
        gw, gb = T.grad(cost, [w, b])

        update = theano.function(
                inputs=[x, y],
                outputs=[prediction, xent],
                updates=[(w, w-0.01 * gw), (b, b - 0.01 * gb)],
                name = "update"
        )
        predict = theano.function(inputs=[x], outputs=prediction, name = "predict")
        for i in range(self.training_steps):
            #testResult = testFun(self.input_data)
            pred, err = update(self.input_data, self.input_label)
            #print(np.mean(predict(self.input_data) == self.input_label, axis = 1))
        #print("after training")
        #




