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

class Adaption(Layer):

    def __init__(self, kernel_size, ndoa, activation=None):

        super(Adaption,self).__init__()
        self.kernel_size = kernel_size
        self.ndoa = ndoa
        self.activation = activations.get(activation)


    def build(self, input_shapes):

        x_shape = input_shapes[0]                                   # shape = (nbatch, samples, nmic)
        pid_shape = input_shapes[1]                                 # shape = (nbatch,)
        self.samples = x_shape[1]
        self.nmic = x_shape[2]

        kernel_shape = (self.ndoa, self.kernel_size, self.nmic, self.nmic)
        self.U = self.add_weight(name='U', shape=kernel_shape, initializer='random_normal', dtype=tf.float32)

        super(Adaption, self).build(input_shapes)


    def call(self, inputs):

        x = inputs[0]                                               # shape = (nbatch, samples, nmic)
        pid = inputs[1]                                             # shape = (nbatch,)

        x = tf.transpose(x, [0,2,1])                                # shape = (nbatch, nmic, samples)
        Fx = tf.signal.rfft(x, fft_length=[self.samples])

        A = tf.one_hot(pid, depth=self.ndoa, on_value=1.0, off_value=0.0, dtype=tf.float32)         # shape = (nbatch, ndoa)
        u = tf.einsum('skmn,bs->bmnk', self.U, A)                                                   # shape = (nbatch, nmic, nmic, kernel_size)
        Fu = tf.signal.rfft(u, fft_length=[self.samples])

        Fy = tf.reduce_sum(Fu*Fx[:,:,tf.newaxis,:], axis=1)
        y = tf.signal.irfft(Fy)
        y = tf.transpose(y, [0,2,1])                                # shape = (nbatch, samples, nmic)

        if self.activation is not None:
            y = self.activation(y)

        return y


    def compute_output_shape(self, input_shapes):

        x_shape = input_shapes[0]                                   # shape = (nbatch, samples, nmic)
        pid_shape = input_shapes[1]                                 # shape = (nbatch,)

        return x_shape


    def get_config(self):

        return super(Adaption,self).get_config()





#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class beamforming(object):

    def __init__(self, fgen):

        self.samples = fgen.samples                     # number of samples per utterance
        self.nmic = fgen.nmic                           # number of microphones
        self.ndoa = fgen.ndoa                           # number of DOA vectors on the sphere

        self.nbin = 500                                 # latent space H
        self.wlen = 200                                 # convolution kernel filter length
        self.shift = self.wlen//4                       # convolution stride

        self.create_model()



    #---------------------------------------------------------
    def forward(self, inp):

        x = inp                                                     # shape = (nbatch, samples, nmic)

        X = tf.signal.frame(x, self.wlen, self.shift, axis=1)       # shape = (nbatch, nfram, wlen, nmic)

        nbatch = tf.shape(X)[0]
        nfram = tf.shape(X)[1]
        X = tf.reshape(X, [nbatch, nfram, self.wlen*self.nmic])

        return X



    #---------------------------------------------------------
    def inverse(self, inp):

        X = inp                                                     # shape = (nbatch, nfram, wlen)

        x = tf.signal.overlap_and_add(X, self.shift)                # shape = (nbatch, nsamples)

        return x



    #---------------------------------------------------------
    def bf_filter(self, inp):

        H = inp[0]                                                                  # shape = (nbatch, nfram, nbin)
        G = inp[1]                                                                  # shape = (nbatch, nfram, nbin)

        Y = H*G
        Py = Y*Y

        return [Y, Py]


    #---------------------------------------------------------
    def cost(self, inp):
        
        r = inp[0]                                                                  # shape = (nbatch, nsamples)
        y = inp[1]                                                                  # shape = (nbatch, nsamples)

        r -= tf.reduce_mean(r, axis=-1, keepdims=True)
        y -= tf.reduce_mean(y, axis=-1, keepdims=True)

        alpha = tf.reduce_sum(r*y, axis=-1) / (tf.reduce_sum(r*r, axis=-1) + 1e-6)
        r *= alpha[:,tf.newaxis]

        Pt = tf.reduce_sum(r*r, axis=-1) + 1e-6
        Pe = tf.reduce_sum((r-y)**2, axis=-1) + 1e-6
        si_sdr = 10*log10(Pt) - 10*log10(Pe)

        cost = -tf.reduce_mean(si_sdr)

        return cost



    #---------------------------------------------------------
    def create_model(self):

        Z = Input(shape=(self.samples, self.nmic), dtype=tf.float32)                # shape = (nbatch, nsamples, nmic)
        R = Input(shape=(self.samples,), dtype=tf.float32)                          # shape = (nbatch, nsamples)
        pid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch,)

        # statistic adaption
        X = Adaption(kernel_size=self.wlen, ndoa=self.ndoa)([Z, pid[:,0]])          # shape = (nbatch, nsamples, nmic)
        # fast convolution
        X = Lambda(self.forward)(X)                                                 # shape = (nbatch, nfram, wlen*nmic)
        H = Dense(units=self.nbin, activation='linear')(X)                          # shape = (nbatch, nfram, nbin)
       
        # beamforming
        X = LayerNormalization()(H)
        X = Bidirectional(LSTM(units=self.nbin, activation='tanh', return_sequences=True))(X)
        X = Dense(units=self.nbin, activation='tanh')(X)                            # shape = (nbatch, nfram, nbin)
        G = Dense(units=self.nbin, activation='linear')(X)                          # shape = (nbatch, nfram, nbin)
        Y, Py = Lambda(self.bf_filter)([H,G])

        # fast deconvolution
        Y = Dense(units=self.wlen, activation='linear')(Y)                         # shape = (nbatch, nfram, wlen)
        Y = Lambda(self.inverse)(Y)                                                # shape = (nbatch, samples)
        cost = Lambda(self.cost)([R, Y])

        self.model = Model(inputs=[Z, R, pid], outputs=[Py, Y, cost])



    #---------------------------------------------------------
    def si_sdr(self, s, y):

        # shape = (nbatch, samples)
        samples = np.minimum(s.shape[-1], y.shape[-1])
        s = s[:,:samples]
        y = y[:,:samples]

        s -= np.mean(s, axis=-1, keepdims=True)
        y -= np.mean(y, axis=-1, keepdims=True)

        alpha = np.sum(y*s, axis=-1) / (np.sum(s*s, axis=-1) + 1e-6)
        s *= alpha[:, np.newaxis]
        Ps = np.sum(s**2, axis=-1) + 1e-6
        Pn = np.sum((s-y)**2, axis=-1) + 1e-6

        si_sdr = 10*np.log10(Ps) - 10*np.log10(Pn)

        return np.mean(si_sdr)




