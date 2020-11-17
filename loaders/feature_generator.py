# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import glob
import argparse
import json
import os
import sys
import numpy as np
from multiprocessing import Pool

sys.path.append(os.path.abspath('../'))

from loaders.respeaker_array import respeaker_array
from loaders.audio_loader import audio_loader
from loaders.rir_loader import rir_loader
from loaders.doa_bases import doa_bases
from algorithms.audio_processing import *
from utils.mat_helpers import *




#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
class feature_generator(object):


    #--------------------------------------------------------------------------
    def __init__(self, config, set='train'):

        self.fs = config['fs']
        self.samples = int(self.fs*config['duration'])
        self.set = set

        self.mic_array = respeaker_array(config)
        self.nmic = self.mic_array.nmic

        self.audio_loader = audio_loader(config, set)
        self.nspk = self.audio_loader.n_speakers

        self.doa_bases = doa_bases(config)
        self.ndoa = self.doa_bases.ndoa

        self.rir_loader = rir_loader(config)
        self.rir_loader.generate_sets(self.doa_bases.doa_idx, self.ndoa)



    #---------------------------------------------------------
    def generate_triplet_indices(self, speakers=20, utterances_per_speaker=3):

        sid = np.random.choice(self.nspk, size=speakers, replace=False).astype(np.int32)
        sid = np.repeat(sid, utterances_per_speaker)

        return sid



    #---------------------------------------------------------
    def generate_multichannel_mixture(self, nsrc=2, sid=None, pid=None, analytic_adaption=False):

        if sid is None:
            # choose nsrc random sid
            sid = np.random.choice(self.nspk, size=nsrc, replace=False).astype(np.int32)
        else:
            # choose nsrc-1 random sid, that are not equal to sid0
            sid0 = sid
            sid1 = [s for s in range(self.nspk) if s != sid0]
            sid1 = np.random.choice(sid1, size=nsrc-1, replace=False).astype(np.int32)
            sid = np.concatenate([[sid0], sid1])

        # generate multichannel signal from WSJ0 and RIR
        Fs = self.audio_loader.load_random_files_from_speaker_ids(sid)             # shape = (nsrc, samples/2+1)
        Fh, Fh_ref, pid = self.rir_loader.load_rirs(nsrc, pid, self.set)           # shape = (nsrc, samples/2+1, nmic)

        Fs = Fs[:,:,np.newaxis]
        Fz = Fs*Fh
        Fr = Fs*Fh_ref

        # mixture: sum of all sources
        Fz = np.sum(Fz, axis=0)                                 # shape = (samples/2+1, nmic)
        # reference: chose source 1, microphone 1
        Fr = Fr[0,:,0]                                          # shape = (samples/2+1,)

        if analytic_adaption is True:
            Fz = self.mic_array.whiten_data(Fz)
            Fv = self.doa_bases.Fv
            Fv = normalize_phase(Fv)[pid[0],:,:]                # adapt to the first pid
            Fz *= np.conj(Fv)

        z = irfft(Fz, n=self.samples, axis=0)                   # shape = (samples, nmic)
        r = irfft(Fr, n=self.samples, axis=0)                   # shape = (samples,)

        return z, r, sid, pid



    #---------------------------------------------------------
    def generate_multichannel_mixtures(self, nsrc=2, sid=[None], pid=None, analytic_adaption=False):

        nbatch = len(sid)
        
        Bz = np.zeros(shape=(nbatch, self.samples, self.nmic), dtype=np.float32)
        Br = np.zeros(shape=(nbatch, self.samples), dtype=np.float32)
        Bsid = np.zeros(shape=(nbatch, nsrc), dtype=np.int32)
        Bpid = np.zeros(shape=(nbatch, nsrc), dtype=np.int32)
        for b in np.arange(nbatch):
            Bz[b,:,:], Br[b,:], Bsid[b,:], Bpid[b,:] = self.generate_multichannel_mixture(nsrc, sid[b], pid, analytic_adaption)

        return Bz, Br, Bsid, Bpid



    #---------------------------------------------------------
    def generate_singlechannel_signals(self, nbatch=10, sid=None, pid=None):

        if sid is None:
            sid = np.random.choice(self.nspk, size=nbatch, replace=True).astype(np.int32)

        # generate multichannel signal from WSJ0 and RIR
        Fs = self.audio_loader.load_random_files_from_speaker_ids(sid)                  # shape = (nbatch, samples/2+1)
        Fh, Fh_ref, pid = self.rir_loader.load_rirs(nbatch, pid, replace=True)          # shape = (nbatch, samples/2+1, nmic)

        Fr = Fs*Fh_ref[:,:,0]                       # reference: chose microphone 1

        r = irfft(Fr, n=self.samples, axis=1)       # shape = (nbatch, samples)

        return r, sid




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='mcss test')
    parser.add_argument('--config_file', help='name of json configuration file', default='../experiments/shoebox_c1.json')
    args = parser.parse_args()


    with open(args.config_file, 'r') as f:
        config = json.load(f)


    fgen = feature_generator(config, set='train')

    t0 = time.time()
    z, r, sid, pid = fgen.generate_multichannel_mixture(nsrc=2, analytic_adaption=True)
    t1 = time.time()
    print(t1-t0)


    data = {
            'z': z,
            'r': r,
           }
    save_numpy_to_mat('fgen_check.mat', data)




