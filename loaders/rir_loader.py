# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import glob
import sys
import os
import json
import argparse
import numpy as np
import pyroomacoustics as pra

sys.path.append(os.path.abspath('../'))
from loaders.respeaker_array import respeaker_array
from algorithms.audio_processing import *
from utils.mat_helpers import *



class rir_loader(object):

    # --------------------------------------------------------------------------
    def __init__(self, config, set='train'):

        self.mic_array = respeaker_array(config)
        self.nmic = self.mic_array.nmic
        self.fs = config['fs']
        self.samples = int(self.fs*config['duration'])
        self.rir_type = config['rir_type']
        self.rir_file = '../data/rir_'+self.rir_type+'.mat'
        self.set = set

        self.cache_rirs()


    #----------------------------------------------------------------------------
    # load rirs from mat file
    def cache_rirs(self):

        if os.path.isfile(self.rir_file):
            data = load_numpy_from_mat(self.rir_file)
            h = data['z']                                               # shape = (nrir, samples, nmic)
            self.nrir = h.shape[0]
        else:
            print(self.rir_file, 'not found.')
            return

        # normalize amplitude for each rir
        mean_amplitude = np.linalg.norm(h, axis=1)                      # shape = (nrir, nmic)
        mean_amplitude = np.mean(mean_amplitude, axis=1)                # shape = (nrir)
        h /= mean_amplitude[:,np.newaxis,np.newaxis]
        self.Fh = rfft(h, n=self.samples, axis=1)                       # shape = (nrir, samples/2+1, nmic)

        print('*** loaded', self.nrir, 'RIRs from:', self.rir_file)

        # generate reference rirs with shortened RT60
        rt60 = 0.050
        b = -np.log(1e-3)/rt60
        tvect = np.arange(h.shape[1])/self.fs
        g = np.exp(-b*tvect)
        h_ref = h*g[np.newaxis,:,np.newaxis]                            # shape = (nrir, samples, nmic)
        h_ref /= np.linalg.norm(h_ref, axis=1, keepdims=True)
        self.Fh_ref = rfft(h_ref, n=self.samples, axis=1)               # shape = (nrir, samples/2+1, nmic)



    #----------------------------------------------------------------------------
    # generate train/test sets using the DOA index
    def generate_sets(self, doa_idx, ndoa):

        # DOA of RIR(i) = doa_idx(i)

        self.rir_list_train = []
        self.rir_list_test = []
        for d in range(ndoa):

            idx = np.where(doa_idx==d)[0]                  # index of RIRs with DOA==d

            if len(idx) > 2:                               # put RIR in train set if the DOA has more than 0 examples
                self.rir_list_train.append(idx[:-1])
                self.rir_list_test.append(idx[-1])         # put RIR in test set if the DOA has more than 2 examples
            elif len(idx) > 0:
                self.rir_list_train.append(idx)


        print('*** using', len(self.rir_list_train), '/', ndoa, 'DOAs for training')
        print('*** using', len(self.rir_list_test), '/', ndoa, 'DOAs for testing')



    #----------------------------------------------------------------------------
    def load_rirs(self, nsrc, pid=None, replace=False):

        if self.set=='train':
            rir_list = self.rir_list_train
        elif self.set=='test':
            rir_list = self.rir_list_test
        else:
            print('Unknown set name:', set)
            quit(0)

        if pid is None:
            pid = np.random.choice(len(rir_list), size=nsrc, replace=replace)                 # loading random DOA indices

        Fh = np.zeros((nsrc, self.samples//2+1, self.nmic), dtype=np.complex64)
        Fh_ref = np.zeros((nsrc, self.samples//2+1, self.nmic), dtype=np.complex64)
        for n in range(nsrc):
            rir_idx = np.random.choice(rir_list[pid[n]])
            Fh[n,:,:] = self.Fh[rir_idx,:,:]
            Fh_ref[n,:,:] = self.Fh_ref[rir_idx,:,:]

        return Fh, Fh_ref, pid



    #----------------------------------------------------------------------------
    # generate a shoebox RIR at a random position in a random room
    def generate_shoebox_rir(self,):

        # match rt60 of the recorded RIRs
        rt60 = np.random.uniform(0.2, 0.3)*0.6       # 0.6 = compensation factor to match true rt60

        # define random room dimensions
        x = np.random.uniform(3,6)
        y = np.random.uniform(3,6)
        z = 3
        room_dim = np.asarray([x,y,z])

        # invert Sabine's formula to obtain the parameters for the ISM simulator
        absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        # create the room
        room = pra.ShoeBox(room_dim, fs=self.fs, materials=pra.Material(absorption), max_order=max_order)

        # place the array randomly in the room, with a minimum distance of <delta> off the walls
        delta = 0.5
        x = np.random.uniform(delta,room_dim[0]-delta)
        y = np.random.uniform(delta,room_dim[1]-delta)
        z = 0.8
        array_center = np.asarray([x,y,z])
        pos = self.mic_array.micpos.T + array_center[:,np.newaxis]
        room.add_microphone_array(pos)

        # place random sources in the room, with a minimum distance of <delta> off the walls
        delta = 0.1
        x = np.random.uniform(delta,room_dim[0]-delta)
        y = np.random.uniform(delta,room_dim[1]-delta)
        z = np.random.uniform(delta,room_dim[2]-delta)
        room.add_source([x, y, z], signal=0, delay=0)

        # compute rir and extend length to 1s
        room.compute_rir()
        samples = int(self.fs*1.0)
        h = np.zeros((samples,self.nmic), dtype=np.float32)
        for m in range(self.nmic):
            h0 = room.rir[m][0]
            if h0.size<samples:
                h0 = np.concatenate([h0, np.zeros((samples-h0.size,), dtype=np.float32)])
            else:
                h0 = h0[:samples]
            h[:,m] = h0

        return h



    #----------------------------------------------------------------------------
    def generate_shoebox_rirs(self, nrir):

        h = []
        for r in range(nrir):
            h.append(self.generate_shoebox_rir())
            print('Generating shoebox RIR', r, '/', nrir)

        h = np.stack(h, axis=0)

        data = {
                'z': h,
               }
        if 'shoebox' in self.rir_file:
            save_numpy_to_mat(self.rir_file, data)




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='respeaker array')
    parser.add_argument('--config_file', help='name of json configuration file', default='../experiments/shoebox_c2.json')
    args = parser.parse_args()


    with open(args.config_file, 'r') as f:
        config = json.load(f)



    rir_loader = rir_loader(config)
    rir_loader.generate_shoebox_rirs(720)



