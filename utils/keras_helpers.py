# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import sys
import time
import h5py

import keras.backend as K
import tensorflow as tf




#-----------------------------------------------------
class Logger(tf.keras.callbacks.Callback):

    def __init__(self, name):
        self.name = name
        self.loss = []
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
        self.loss = []

    def on_batch_end(self, batch, logs=None):
        if np.isnan(np.sum(logs['loss'])):
            quit()
        self.loss = np.append(self.loss, logs['loss'])
        #print('end of batch: ', logs['loss'].shape)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time_end = time.time()
        duration = self.epoch_time_end-self.epoch_time_start
        self.epoch += 1

        print('model:', self.name, ', epoch:', self.epoch, ', runtime:', '{0:.3f}'.format(duration), ', loss:', '{0:.3f}'.format(np.mean(self.loss)) )



#-----------------------------------------------------
def log10(x):

    return tf.math.log(x) / 2.302585092994046



#---------------------------------------------------------
def batch_sum(x, include_batch_dimension=False):

    if include_batch_dimension is False:
        all_but_first_axes = tuple(range(1, K.ndim(x)))
        return tf.reduce_sum(x, axis=all_but_first_axes)
    else:
        return tf.reduce_sum(x)


#---------------------------------------------------------
def batch_mean(x, include_batch_dimension=False):

    if include_batch_dimension is False:
        all_but_first_axes = tuple(range(1, K.ndim(x)))
        return tf.reduce_mean(x, axis=all_but_first_axes)
    else:
        return tf.reduce_mean(x)


#---------------------------------------------------------
def weighted_mse(p_true, p_est, weight=None):

    mse = (p_true-p_est)**2

    if weight is None:
        return batch_mean(mse)
    else:
        return batch_sum(mse*weight) / batch_sum(weight)


#-----------------------------------------------------
def weighted_bce(p_true, p_est, weight=None, include_batch_dimension=False):

    eps = 1e-6
    p_true = tf.clip_by_value(p_true, eps, 1.0-eps)
    p_est = tf.clip_by_value(p_est, eps, 1.0-eps)
    bce = p_true*tf.math.log(p_est) + (1-p_true)*tf.math.log(1-p_est)

    if weight is None:
        return -batch_mean(bce, include_batch_dimension)
    else:
        return -batch_sum(bce*weight, include_batch_dimension) / batch_sum(weight, include_batch_dimension)


#-----------------------------------------------------
def weighted_cce(p_true, p_est, weight=None, axis=-1):

    p_true = tf.clip_by_value(p_true, 1e-6, 1.0)
    p_est = tf.clip_by_value(p_est, 1e-6, 1.0)
    cce = tf.reduce_sum(p_true*tf.math.log(p_est), axis=-1)

    if weight is None:
        return -batch_mean(cce)
    else:
        return -batch_sum(cce*weight) / batch_sum(weight)


