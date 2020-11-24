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

from modules.beamforming_fd import beamforming
from modules.identification_fd import identification


np.set_printoptions(precision=3, threshold=3, edgeitems=3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class bssd(object):

    def __init__(self, config, set='train'):

        self.config = config
        self.fgen = feature_generator(config, set)
        self.nsrc = config['nsrc']                      # number of concurrent speakers

        self.filename = os.path.basename(__file__)
        self.name = self.filename[:-3] + '_' + config['rir_type']
        self.creation_date = os.path.getmtime(self.filename)
        self.weights_file = self.config['weights_path'] + self.name + '.h5'
        self.predictions_file = self.config['predictions_path'] + self.name + '.mat'
        self.logger = Logger(self.name)

        self.samples = self.fgen.samples                # number of samples per utterance
        self.nmic = self.fgen.nmic                      # number of microphones
        self.ndoa = self.fgen.ndoa                      # number of DOA vectors on the sphere

        self.wlen = 1024                                # FFT length
        self.nbin = self.wlen//2 + 1                    # number of frequency bins
        self.shift = self.wlen//4                       # STFT shift
        self.ndim = 100                                 # embedding dimension E

        self.beamforming = beamforming(self.fgen)
        self.identification = identification(self.fgen)
        self.create_model()

        self.si_sdr = []
        self.eer = []
        self.epoch = 0
        data = load_numpy_from_mat(self.predictions_file)
        if data is not None:
            if 'epoch' in data.keys():
                self.epoch = data['epoch']
                self.si_sdr = data['si_sdr']
                self.eer = data['eer']



    #---------------------------------------------------------
    def create_model(self):

        print('*** creating model: %s' % self.name)

        Z = Input(shape=(self.samples, self.nmic), dtype=tf.float32)                # shape = (nbatch, nsamples, nmic)
        R = Input(shape=(self.samples,), dtype=tf.float32)                          # shape = (nbatch, nsamples)
        pid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch,)
        sid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch, 1)

        [Py, Y, cost_bf] = self.beamforming.model([Z, R, pid])
        [E, cost_id] = self.identification.model([Py, sid])

        # compile model
        self.model = Model(inputs=[Z, R, pid, sid], outputs=[Y, E])
        self.model.add_loss(cost_bf + 0.01*cost_id)
        self.model.compile(loss=None, optimizer='adam')

        print(self.model.summary())
        try:
            self.model.load_weights(self.weights_file)
        except:
            print('error loading weights file: %s' % self.weights_file)



    #---------------------------------------------------------
    def save_weights(self):

        self.model.save_weights(self.weights_file)

        return



    #---------------------------------------------------------
    def train(self):

        print('train the model')
        while (self.epoch<self.config['epochs']) and self.check_date():

            sid0 = self.fgen.generate_triplet_indices(speakers=20, utterances_per_speaker=3)
            z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid0)
            self.model.fit([z, r, pid[:,0], sid[:,0]], None, batch_size=len(sid0), epochs=1, verbose=0, shuffle=False, callbacks=[self.logger])

            self.epoch += 1
            if (self.epoch%100)==0:
                self.save_weights()
                self.validate()



    #---------------------------------------------------------
    def validate(self):

        sid = self.fgen.generate_triplet_indices(speakers=self.fgen.nspk, utterances_per_speaker=3)
        z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid)
        y, E = self.model.predict([z, r, pid[:,0], sid[:,0]], batch_size=50)

        si_sdr = self.beamforming.si_sdr(r, y)
        far, frr, eer = self.identification.calc_eer(E, sid[:,0])
        print('SI-SDR:', si_sdr)
        print('EER:', eer)
        self.si_sdr = np.append(self.si_sdr, si_sdr)
        self.eer = np.append(self.eer, eer)

        data = {
            'z': z[0,:,0],
            'r': r[0,:],
            'y': y[0,:],
            'E': E,
            'pid': pid,
            'sid': sid,
            'far': far,
            'frr': frr,
            'si_sdr': self.si_sdr,
            'eer': self.eer,
            'epoch': self.epoch,
        }
        save_numpy_to_mat(self.predictions_file, data)



    #---------------------------------------------------------
    def plot(self):

        z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc)
        
        data = []
        z0 = z[0,:,0]/np.amax(np.abs(z[0,:,0]))
        data.append( 20*np.log10(np.abs(mstft(z0))) )

        for c in range(self.nsrc):
            y, E = self.model.predict([z, r, pid[:,c], sid[:,c]])
            y0 = y[0,:]/np.amax(np.abs(y[0,:]))
            data.append( 20*np.log10(np.abs(mstft(y0))) )

        legend = ['mixture z(t)', 'extracted speaker y1(t)', 'extracted speaker y2(t)', 'extracted speaker y3(t)', 'extracted speaker y4(t)']
        filename = self.config['predictions_path'] + self.name + '_spectrogram.png'
        draw_subpcolor(data, legend, filename)



    #---------------------------------------------------------
    def check_date(self):

        if (self.creation_date == os.path.getmtime(self.filename)):
            return True
        else:
            return False





#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    # parse command line args
    parser = argparse.ArgumentParser(description='speaker separation')
    parser.add_argument('--config_file', help='name of json configuration file', default='shoebox_c2.json')
    parser.add_argument('mode', help='mode: [train, valid, plot]', nargs='?', choices=('train', 'valid', 'plot'), default='train')
    args = parser.parse_args()


    # load config file
    try:
        print('*** loading config file: %s' % args.config_file )
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except:
        print('*** could not load config file: %s' % args.config_file)
        quit(0)



    if args.mode == 'train':
        bssd = bssd(config)
        bssd.train()

    if args.mode == 'valid':
        bssd = bssd(config)
        bssd.validate()

    if args.mode == 'plot':
        bssd = bssd(config)
        bssd.plot()


