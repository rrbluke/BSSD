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

    def __init__(self, ndoa, activation=None):

        super(Adaption,self).__init__()
        self.ndoa = ndoa
        self.activation = activation


    def build(self, input_shapes):

        X_shape = input_shapes[0]         # shape = (nbatch, nfram, nbin, nmic)
        pid_shape = input_shapes[1]       # shape = (nbatch,)
        nbin = X_shape[-2]
        nmic = X_shape[-1]

        self.U_real = self.add_weight(name='U_real', shape=(self.ndoa, nbin, nmic, nmic), initializer='random_normal', dtype=tf.float32)
        self.U_imag = self.add_weight(name='U_imag', shape=(self.ndoa, nbin, nmic, nmic), initializer='random_normal', dtype=tf.float32)

        super(Adaption, self).build(input_shapes)


    def call(self, inputs):

        X = inputs[0]                    # shape = (nbatch, nfram, nbin, nmic)
        pid = inputs[1]                  # shape = (nbatch,)

        X = tf.cast(X, tf.complex64)

        A = tf.one_hot(pid, depth=self.ndoa, on_value=1.0, off_value=0.0, dtype=tf.float32)         # shape = (nbatch, ndoa)
        U_real = tf.einsum('skmn,bs->bkmn', self.U_real, A)                                         # shape = (nbatch, nbin, nmic, nmic)
        U_imag = tf.einsum('skmn,bs->bkmn', self.U_imag, A)                                         # shape = (nbatch, nbin, nmic, nmic)
        U = cast_to_complex(U_real, U_imag)
        Y = einsum('btkm,bkmn->btkn', X, U)

        #if self.activation is not None:

        return Y


    def compute_output_shape(self, input_shapes):

        X_shape = input_shapes[0]         # shape = (nbatch, nfram, nbin, nmic)
        pid_shape = input_shapes[1]       # shape = (nbatch,)

        return X_shape


    def get_config(self):

        return super(Adaption,self).get_config()





#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class beamforming(object):

    def __init__(self, fgen):

        self.samples = fgen.samples                     # number of samples per utterance
        self.nmic = fgen.nmic                           # number of microphones
        self.ndoa = fgen.ndoa                           # number of DOA vectors on the sphere

        self.wlen = 1024                                # FFT length
        self.nbin = self.wlen//2 + 1                    # number of frequency bins
        self.shift = self.wlen//4                       # STFT shift

        self.create_model()



    #---------------------------------------------------------
    def forward(self, inp):

        x = inp                                                                     # shape = (nbatch, samples, nmic)

        x = tf.transpose(x, [0,2,1])                                                # shape = (nbatch, nmic, samples)
        X = tf.signal.stft(x, frame_length=self.wlen, frame_step=self.shift)        # shape = (nbatch, nmic, nfram, nbin)
        X = tf.transpose(X, [0,2,3,1])                                              # shape = (nbatch, nfram, nbin, nmic)

        nbatch = tf.shape(X)[0]
        nfram = tf.shape(X)[1]

        return X



    #---------------------------------------------------------
    def inverse(self, inp):

        X = inp                                                                      # shape = (nbatch, nfram, nbin)

        x = tf.signal.inverse_stft(X, frame_length=self.wlen, frame_step=self.shift) # shape = (nbatch, nsamples)

        return x



    #---------------------------------------------------------
    def normalize(self, inp):

        Fz = tf.cast(inp, tf.complex64)                                             # shape = (nbatch, nfram, nbin, nmic)
        nbatch = tf.shape(Fz)[0]
        nfram = tf.shape(Fz)[1]

        vz = vector_normalize_magnitude(Fz)                                         # shape = (nbatch, nfram, nbin, nmic)
        vz = vector_normalize_phase(vz)                                             # shape = (nbatch, nfram, nbin, nmic)

        Z = cast_to_float(vz)                                                       # shape = (nbatch, nfram, nbin, nmic, 2)
        Z = tf.reshape(Z, [nbatch, nfram, self.nbin*self.nmic*2])                   # shape = (nbatch, nfram, nbin*nmic*2)

        return Z



    #---------------------------------------------------------
    def bf_filter(self, inp):

        H = tf.cast(inp[0], tf.complex64)                                           # shape = (nbatch, nfram, nbin, nmic)
        W = inp[1]                                                                  # shape = (nbatch, nfram, nbin*nmic*2)
        nbatch = tf.shape(H)[0]
        nfram = tf.shape(H)[1]

        W = tf.reshape(W, [nbatch, nfram, self.nbin, self.nmic, 2])
        W = cast_to_complex(W[...,0], W[...,1])

        Fy = einsum('btkm,btkm->btk', H, W)                                         # shape = (nbatch, nfram, nbin)
        Py = elementwise_abs2(Fy)

        return [Fy, Py]



    #---------------------------------------------------------
    def cost(self, inp):
        
        r = inp[0]                                                                  # shape = (nbatch, nsamples)
        y = inp[1]                                                                  # shape = (nbatch, nsamples)

        samples = self.samples-self.wlen
        r = r[:,:samples]
        y = y[:,:samples]

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

        # STFT
        X = Lambda(self.forward)(Z)                                                 # shape = (nbatch, nfram, nbin, nmic)
        # statistic adaption
        H = Adaption(ndoa=self.ndoa, activation=None)([X, pid[:,0]])                # shape = (nbatch, nfram, nbin, nmic)

        # beamforming
        X = Lambda(self.normalize)(H)                                               # shape = (nbatch, nfram, nbin*nmic*2)
        X = Dense(units=self.nbin, activation='tanh')(X)
        X = Bidirectional(LSTM(units=self.nbin, activation='tanh', return_sequences=True))(X)
        X = Dense(units=self.nbin, activation='tanh')(X)
        W = Dense(units=self.nbin*self.nmic*2, activation='linear')(X)
        Fy, Py = Lambda(self.bf_filter)([H,W])                                      # shape = (nbatch, nfram, nbin)

        # iSTFT
        Y = Lambda(self.inverse)(Fy)                                                # shape = (nbatch, samples)
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




