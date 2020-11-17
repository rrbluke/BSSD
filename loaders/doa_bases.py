# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import numpy as np
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath('../'))
from loaders.respeaker_array import respeaker_array
from loaders.rir_loader import rir_loader
from algorithms.audio_processing import *
from utils.mat_helpers import *


np.set_printoptions(precision=3, threshold=10, edgeitems=10)





#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class doa_bases(object):

    def __init__(self, config):

        self.config = config
        self.mic_array = respeaker_array(config)
        self.name = 'doa_bases'
        self.rir_type = config['rir_type']
        self.doa_file = '../data/doa_'+self.rir_type+'.mat'

        self.fs = self.mic_array.fs
        self.samples = int(self.fs*config['duration'])
        self.nmic = self.mic_array.nmic


        self.Fv = self.generate_doa_bases(nbin=self.samples//2+1)
        print('*** generated', self.ndoa, 'DOA positions on a sphere')

        if os.path.isfile(self.doa_file):
            data = load_numpy_from_mat(self.doa_file)
            self.doa_idx = np.squeeze(data['doa_idx'])                        # shape = (nrir,)
        else:
            print('DOA file:', self.doa_file, 'not found')



    #----------------------------------------------------------------------------
    # generate a anechoic rir from a point on a sphere
    def generate_doa_vector(self, phi, theta, rad, nbin):

        # phi      -pi ... pi
        # theta     0 ... pi/2
        # rad       > 0

        xyz = np.zeros((3,), dtype=np.float32)
        xyz[0] = rad*np.cos(theta)*np.cos(phi)
        xyz[1] = rad*np.cos(theta)*np.sin(phi)
        xyz[2] = rad*np.sin(theta)
        dist = xyz[np.newaxis,:] - self.mic_array.micpos                      # shape = (nmic, 3)
        dist = np.linalg.norm(dist, axis=-1)                                  # shape = (nmic,)
        tau = dist/self.mic_array.c
        fvect = np.linspace(0, self.fs*0.5, nbin, endpoint=True)
        arg = np.einsum('m,k->km', tau, fvect)                                # shape = (nbin, nmic)
        Fv = np.exp(-1j*2*np.pi*arg)

        return Fv



    #---------------------------------------------------------
    def generate_doa_bases(self, nbin):

        self.ndoa = 100
        self.phi_range = np.zeros((self.ndoa,), dtype=np.float32)
        self.theta_range = np.zeros((self.ndoa,), dtype=np.float32)

        # generate a Fibonacci Spiral Sphere
        gr = (np.sqrt(5)+1)/2         # golden ratio = 1.618
        ga = (2-gr)*(2*np.pi)         # golden angle = 2.399
        for n in range(self.ndoa):
            self.theta_range[n] = np.arcsin(n/self.ndoa)
            self.phi_range[n] = ga*n

        Fv = np.zeros((self.ndoa, nbin, self.nmic), dtype=np.complex64)                      # shape = (ndoa, nbin, nmic)
        for i in range(self.ndoa):
            Fv[i,:,:] = self.generate_doa_vector(phi=self.phi_range[i], theta=self.theta_range[i], rad=1.0, nbin=nbin)

        Fv = self.mic_array.whiten_data(Fv)                                                  # shape = (ndoa, nbin, nmic)

        return Fv



    #-------------------------------------------------------------------------
    # normalize phase of Fz relative to the first microphone
    def normalize_phase(self, Fz):

        phi = np.conj(Fz[...,0]) / (np.abs(Fz[...,0]) + 1e-6)
        Fu = Fz * phi[...,np.newaxis]

        return Fu


    #-------------------------------------------------------------------------
    # normalize magnitude of Fz to 1 along the nmic axis
    def normalize_magnitude(self, Fz):

        Fu = Fz / (np.linalg.norm(Fz, axis=-1, keepdims=True) + 1e-6)

        return Fu



    #---------------------------------------------------------
    def estimate_doa(self, Fh):

        Fh = self.mic_array.whiten_data(Fh)
        vh = self.normalize_magnitude(Fh)                                                    # shape = (nrir, nbin, nmic)
        nrir = Fh.shape[0]

        vv = self.normalize_magnitude(self.Fv)

        # weighting factor: with 1/f
        fvect = np.linspace(0, self.fs*0.5, self.samples//2+1, endpoint=True)
        w = np.minimum(1000/(fvect+1e-6), 1)

        p = np.zeros((nrir, self.ndoa), dtype=np.float32)
        for r in range(nrir):
            gamma = np.abs(np.einsum('km,dkm->dk', vh[r,:,:], np.conj(vv)))**2              # shape = (ndoa, nbin)
            #p[r,:] = np.mean(1/(1-gamma+1e-3), axis=-1)                                    # shape = (ndoa,)
            p[r,:] = np.sum(gamma*w[np.newaxis,:], axis=-1) / np.sum(w)
            print('calculating DOA estimate for RIR:', r, '/', nrir)

        doa_idx = np.argmax(p, axis=-1)


        data = {
            'p': p,
            'doa_idx': doa_idx,
            'theta_range': self.theta_range,
            'phi_range': self.phi_range,
        }
        save_numpy_to_mat(self.doa_file, data)

        print('Saved DOA index to:', self.doa_file)





#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    # parse command line args
    parser = argparse.ArgumentParser(description='doa bases')
    parser.add_argument('--config_file', help='name of json configuration file', default='../experiments/shoebox_c2.json')
    args = parser.parse_args()


    # load config file
    try:
        print('*** loading config file: %s' % args.config_file )
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except:
        print('*** could not load config file: %s' % args.config_file)
        quit(0)


    rir = rir_loader(config)
    doa = doa_bases(config)
    doa.estimate_doa(rir.Fh)



