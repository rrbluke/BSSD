# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import numpy as np
import argparse
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath('../'))
from keras.models import Sequential, Model
from keras.layers import Layer, Dense, Activation, LSTM, Input, Lambda, BatchNormalization, LayerNormalization, Conv1D, Bidirectional
from keras import activations
import keras.backend as K
import tensorflow as tf
from loaders.feature_generator import feature_generator
from utils.mat_helpers import *
from algorithms.audio_processing import *
from utils.keras_helpers import *
from ops.complex_ops import *
from utils.matplotlib_helpers import *


np.set_printoptions(precision=3, threshold=3, edgeitems=3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)






#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class identification(object):

    def __init__(self, fgen):

        self.wlen = 1024                                # FFT length
        self.nbin = self.wlen//2 + 1                    # number of frequency bins
        self.shift = self.wlen//4                       # STFT shift

        self.ndim = 100                                 # embedding dimension E

        self.create_model()



    #---------------------------------------------------------
    def average_pool(self, inp):

        X = inp                                                                     # shape = (nbatch, nfram, ndim)
        X = tf.reduce_mean(X, axis=1)                                               # shape = (nbatch, ndim)

        return X



    #---------------------------------------------------------
    def residualConv(self, X, kernel_size, dilation_rate=1, activation='softplus'):

        Z = Conv1D(filters=X.shape[-1], kernel_size=kernel_size, activation=activation, padding='same', dilation_rate=dilation_rate)(X)
        Z = LayerNormalization()(Z) + X

        return Z



    #---------------------------------------------------------
    def cost(self, inp):

        E = inp[0]                                                                  # shape = (nbatch, ndim)
        sid = tf.cast(inp[1][:,0], tf.int32)                                        # shape = (nbatch,)
        nbatch = tf.shape(sid)[0]

        # distance matrix
        D = tf.reduce_sum((E[tf.newaxis,:,:] - E[:,tf.newaxis,:])**2, axis=-1)      # shape = (nbatch, nbatch)
        D = tf.sqrt(D+1e-6)

        # positive and negative mask
        indices_not_equal = 1-tf.eye(nbatch)
        labels_equal = tf.cast(tf.equal(sid[tf.newaxis,:], sid[:,tf.newaxis]), tf.float32)
        Mp = indices_not_equal*labels_equal
        Mn = 1-labels_equal

        # get the hardest postive example for each anchor
        anchor_positive_dist = Mp*D
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1)
        #hardest_positive_dist = Debug('hardest_positive_dist', hardest_positive_dist)

        # get the hardest negative example for each anchor
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(D, axis=1, keepdims=True)
        anchor_negative_dist = D + max_anchor_negative_dist * (1 - Mn)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1)
        #hardest_negative_dist = Debug('hardest_negative_dist', hardest_negative_dist)

        margin = 0.2
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        cost_tl = tf.reduce_mean(triplet_loss)

        # regularization: BCE
        E -= tf.reduce_mean(E, axis=-1, keepdims=True)
        E /= tf.norm(E, axis=-1, keepdims=True) + 1e-6
        D = tf.einsum('bm,cm->bc', E, E)**2
        cost_bce = tf.reduce_mean(weighted_bce(1-Mn, D))

        return cost_tl + cost_bce*0.01



    #---------------------------------------------------------
    def create_model(self):

        Py = Input(shape=(None, self.nbin), dtype=tf.float32)                       # shape = (nbatch, nfram, nbin)
        sid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch, 1)

        # identification
        X = Dense(units=self.ndim*2, activation='softplus')(Py)
        X = LayerNormalization(axis=1, center=False, scale=False)(X)
        X = self.residualConv(X, kernel_size=10, dilation_rate=1)
        X = self.residualConv(X, kernel_size=10, dilation_rate=2)
        X = self.residualConv(X, kernel_size=10, dilation_rate=4)
        X = self.residualConv(X, kernel_size=10, dilation_rate=8)
        X = self.residualConv(X, kernel_size=10, dilation_rate=16)
        X = self.residualConv(X, kernel_size=10, dilation_rate=32)
        X = Lambda(self.average_pool)(X)
        E = Dense(units=self.ndim, activation='linear')(X)                         # shape = (nbatch, ndim)
        cost = Lambda(self.cost)([E, sid])

        self.model = Model(inputs=[Py, sid], outputs=[E, cost])



    #---------------------------------------------------------
    def calc_eer(self, E, sid):

        nbatch = sid.shape[0]

        # distance matrix
        D = np.sum((E[np.newaxis,:,:] - E[:,np.newaxis,:])**2, axis=-1)                    # shape = (nbatch, nbatch)
        D = np.sqrt(D+1e-6)

        # positive and negative mask
        indices_not_equal = np.eye(nbatch) < 0.5
        labels_equal = np.equal(sid[np.newaxis,:], sid[:,np.newaxis])
        Mp = np.logical_and(indices_not_equal, labels_equal)
        Mn = np.logical_not(labels_equal)

        N = 500
        thres = np.linspace(np.amin(D), np.amax(D), N)
        FAR = np.zeros((N,), dtype=np.float32)
        FRR = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            FAR[i] = np.mean( D[Mn] < thres[i] )
            FRR[i] = np.mean( D[Mp] > thres[i] )

        i = np.argmin(abs(FAR-FRR))
        EER = FRR[i]

        return FAR, FRR, EER


