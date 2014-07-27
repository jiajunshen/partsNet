from __future__ import division
import math
from scipy.special import logit
import time
from scipy.misc import logsumexp
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
import numpy as np
import random, collections
import scipy.sparse

EPS = np.finfo(float).eps

def log_product_of_bernoullis_mixture_likelihood(X, logit_odds, log_inv_mean_sums):
    """
    X:
    Here the X is the original patche data (with padded window)
    logit_odds:
    The logit parts with background probabilities as well.
    log(mean / 1 - mean)
    log_inv_mean_sums:
    (1 / (log mean)).sum()
    """
    
    
    n, d = X.shape
    m = log_inv_mean_sums.size

    memory_limit = 100

    b = max(math.floor(memory_limit * 1024 * 1024 / (64 * d) - m), 1)
    
    num_b = math.ceil(n / b)

    loglike = np.empty((n,m))
    for i in range(num_b):
        loglike[i * b : (i + 1) * b] = np.dot(X[i * b:(i + 1) * b], logit_odds.transpose()) + log_inv_mean_sums

    return loglike

def Extract(data, num_mixture_component, num_rot, parts_shape, region_shape,shifting_shape, mu, bkg_probability):
    n,d = data.shape
    numShifting = shifting_shape[0] * shifting_shape[1]
    partDimension = parts_shape[0] * parts_shape[1] * parts_shape[2]
    bkgRegion = region_shape[0] * region_shape[1] * region_shape[2] - partDimension
    log_odd = np.empty((num_mixture_component * num_rot,) + region_shape)
    log_q = np.empty((numShifting, n, num_mixture_component * num_rot))
    sum_log_one_mu = np.log(1 - mu).sum(axis = 1) + np.log(1 - bkg_probability) * bkgRegion

    for i in range(shifting_shape[0]):
        for j in range(shifting_shape[1]):
            log_odd[:num_mixture_component * num_rot] = np.ones((num_mixture_component * num_rot,) + region_shape) * bkg_probability
            log_odd[:num_mixture_component * num_rot,i:i+parts_shape[0],j :j+ parts_shape[1],:] = mu.reshape((num_mixture_component * num_rot,) + parts_shape)
            log_odd = logit(log_odd)
            log_q[i * shifting_shape[1] + j] = log_product_of_bernoullis_mixture_likelihood(data, log_odd.reshape((num_mixture_component * num_rot, -1)), sum_log_one_mu)
    norm_log_q = logsumexp(logsumexp(log_q, axis = 2),axis = 0)
    log_q -= norm_log_q.reshape((1,n,1))
    data_m = logsumexp(log_q, axis = 0).argmax(axis = 1)
    idx = np.ravel_multi_index((np.arange(n),data_m),(n,num_mixture_component * num_rot))
    data_p = log_q.reshape((numShifting, n * num_mixture_component * num_rot))[:,idx].argmax(axis = 0)
    data_label = np.hstack((data_m.reshape((-1,1)),data_p.reshape((-1,1))))
    return data_label



def LatentShiftRotationEM(data, num_mixture_component, parts_shape, region_shape, shifting_shape, num_rot, max_num_iteration = 25,loglike_tolerance=1e-3, mu_truncation = (1, 1), additional_mu = None, permutation = None, partPermutation = None, numpy_rng=None, verbose = False):
    inverse_permutation = np.argsort(permutation)
    n,d = data.shape
    partDimension = parts_shape[0] * parts_shape[1] * parts_shape[2]
    numShifting = shifting_shape[0] * shifting_shape[1]
    print(n,d,partDimension,numShifting)
    bkgRegion = region_shape[0] * region_shape[1] * region_shape[2] - partDimension
    if(isinstance(mu_truncation, float)):
        use_epsilon = True
        epsilon = mu_truncation
    else:
        use_epsilon = False
        beta_prior = mu_truncation

   
    purity_level = 2 
    log_w = np.empty((num_rot, numShifting,num_mixture_component))
    log_w.fill(-np.log(num_rot * numShifting * num_mixture_component))
    mu = numpy_rng.uniform(size = (num_mixture_component, num_rot * partDimension)) ** (1 / (purity_level + 1))
    centerXStart = int((shifting_shape[0] - 1)/2)
    centerYStart = int((shifting_shape[1] - 1)/2)
    is_flip = np.logical_not((data.reshape((n,num_rot,)+region_shape)[:,:,centerXStart:centerXStart + parts_shape[0], centerYStart:centerYStart + parts_shape[1],:]).reshape(n,-1)[numpy_rng.choice(n,num_mixture_component,replace=False)])
    print(is_flip.shape)            
    mu[is_flip] = 1 - mu[is_flip]
    bkg_probability = 0.2

    if use_epsilon:
        mu[mu < epsilon] = epsilon
        mu[mu > 1 - epsilon] = 1 - epsilon
    else:
        mu *= (n / num_mixture_component) / ((n / num_mixture_component) + np.sum(beta_prior))
        mu += beta_prior[0] / ((n / num_mixture_component) + np.sum(beta_prior))

    log_odd = np.empty((num_mixture_component,num_rot,)+region_shape)
    log_odd_inner = np.empty((d,num_mixture_component))
    sum_log_one_mu = np.empty(num_mixture_component)
    log_q = np.empty((num_rot, numShifting, n, num_mixture_component))
    print(log_w[:,:,4])
    # DO EM.
    loglike = []
    t = 0
    while t < max_num_iteration:
        if verbose:
            clock_start = time.clock()
        print("E-Step")
        # E -step : Compoute q
        sum_log_one_mu = np.log(1 - mu).sum(axis = 1) + np.log(1 - bkg_probability) * bkgRegion * num_rot
        log_odd_inner = mu.transpose()
        #print(log_odd_inner)
        for rot in range(num_rot):
            for i in range(shifting_shape[0]):
                for j in range(shifting_shape[1]):
                    log_odd[:num_mixture_component] = np.ones((num_mixture_component,num_rot,)+ region_shape) * bkg_probability
                    #print(partPermutation[rot])
                    #log_odd[:num_mixture_component,:,i:i+parts_shape[0],j:j+parts_shape[1],:] = log_odd_inner.transpose().reshape((num_mixture_component,num_rot,) + parts_shape)
                    log_odd[:num_mixture_component,:,i:i+parts_shape[0],j:j+parts_shape[1],:] = log_odd_inner[partPermutation[rot]].transpose().reshape((num_mixture_component,num_rot,) + parts_shape)
                    #log_odd = log_odd.reshape(num_mixture_component,-1)
                    #log_odd = np.swapaxes(log_odd,0,1)[permutation[rot]]
                    #log_odd = np.swapaxes(log_odd,0,1)
                    log_odd = logit(log_odd)
                    #log_odd = log_odd.reshape((num_mixture_component,num_rot,)+region_shape)
                    log_q[rot,i * shifting_shape[1] + j] = log_product_of_bernoullis_mixture_likelihood(data,log_odd.reshape((num_mixture_component, -1)),sum_log_one_mu)
        log_q = log_q + log_w.reshape(num_rot,numShifting,1,num_mixture_component)
        norm_log_q = logsumexp(logsumexp(logsumexp(log_q, axis = 3),axis = 0),axis = 0)
        log_q -= norm_log_q.reshape((1,1,n,1))

        print("M-Step")
        # M - Step: Computer weights and model.
        log_w = logsumexp(log_q, axis = 2)
        log_q_sum_r_n_p = logsumexp(logsumexp(log_w,axis = 0),axis = 0)
        print(logsumexp(log_w))
        log_w -= logsumexp(log_w)
        #print(log_w[:,:,4])
        print(np.sum(np.sum(log_w,axis = 0),axis = 1))
        q = np.exp(log_q)
        mu = np.zeros((num_mixture_component,num_rot * partDimension))
        p_bkg = 0
        for i in range(shifting_shape[0]):
            for j in range(shifting_shape[1]):
                for rot in range(num_rot):
                    dotResult = np.dot(q[rot, i * shifting_shape[1] + j].transpose(),data).reshape((num_mixture_component,-1))
                    dotResult = dotResult[:,inverse_permutation[rot]].reshape((num_mixture_component,num_rot,)+region_shape)
                    mu += dotResult[:,:,i:i+parts_shape[0],j:j+parts_shape[1],:].reshape((num_mixture_component,-1))
                    dotResult[:,:,i:i+parts_shape[0],j:j+parts_shape[1],:] = 0
                    p_bkg += np.sum(dotResult)
        bkg_probability = p_bkg / (n * bkgRegion * num_rot)
        print(bkg_probability)        
        
        if use_epsilon:
            eps = np.finfo(np.float_).eps
            mu[mu<eps] = eps
            mu = np.exp(np.log(mu) - log_q_sum_r_n_p.reshape((num_mixture_component, 1)))
            mu[mu < epsilon] = epsilon
            mu[mu > 1- epsilon] = 1 - epsilon
        else:
            mu += beta_prior[0]
            mu /= (np.exp(log_q_sum_r_n_p) + np.sum(beta_prior)).reshape((num_mixture_component,1))

        loglike.append(norm_log_q.sum())
        if verbose:
            print('Iter {}:{:.3f} seconds. Log-likelihood : {:.1f}'.format(t + 1, time.clock() - clock_start, loglike[-1]))
        #if t >= 1 and loglike[-1] - loglike[-2] < loglike_tolerance *  - loglike[-2]:
            #break
        #    continue
        t+=1
    loglike = np.asarray(loglike,dtype = np.float64)
    log_weight = (log_w.sum(axis = 0)).sum(axis = 0)
    ordering = np.argsort(log_weight)[::-1]
    log_weight = log_weight[ordering]
    mu = mu[ordering]

    data_m = logsumexp(logsumexp(log_q,axis = 0),axis = 0).argmax(axis = 1)
    idx = np.ravel_multi_index((np.arange(n),data_m),(n,num_mixture_component))
    data_p = log_q.reshape((num_rot*numShifting,n * num_mixture_component))[:,idx].argmax(axis = 0)
    data_p = np.unravel_index(data_p,(num_rot,numShifting))
    inverse_ordering = np.argsort(ordering)
    data_m[data_m < num_mixture_component] = inverse_ordering[data_m[data_m < num_mixture_component]]
    data_label = np.hstack((data_m.reshape((-1,1)),data_p[0].reshape((-1,1)),data_p[1].reshape((-1,1))))
    print(bkg_probability)
    print(log_weight)
    if verbose:
        print('latentShiftingEM finished')
    return log_weight, mu, loglike,data_label, bkg_probability

