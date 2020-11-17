# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import glob
import argparse
import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../'))

from algorithms.audio_processing import *




class respeaker_array(object):

    # --------------------------------------------------------------------------
    def __init__(self, config):

        self.fs = config['fs']
        self.samples = int(self.fs*config['duration'])
        self.nmic = 6                         # number of microphones
        self.radius = 46.3/1000               # radius of the respeaker core v2 microphone array
        self.c = 343.0                        # speed of sound at 20°C

        self.mic_aperture()
        self.generate_whitening_matrix()



    #-------------------------------------------------------------------------
    def mic_aperture(self):

        self.micpos = np.zeros((self.nmic, 3))

        #mics 1..6 are on a circle
        for m in np.arange(self.nmic):
            a = -2*np.pi*m/self.nmic                    # microphones are arranged clockwise!
            self.micpos[m,0] = self.radius*np.cos(a)    # x-plane
            self.micpos[m,1] = self.radius*np.sin(a)    # y-plane
            self.micpos[m,2] = 0                        # z-plane



    #----------------------------------------------------------------------------
    def generate_whitening_matrix(self, nbin=513):

        dist = self.micpos[:,np.newaxis,:] - self.micpos[np.newaxis,:,:]        # shape = (nmic, nmic, 3)
        dist = np.linalg.norm(dist, axis=-1)                                    # shape = (nmic, nmic)
        tau = dist/self.c

        self.U = np.zeros((nbin, self.nmic, self.nmic), dtype=np.complex64)     # whitening matrix
        for k in range(nbin):

            fc = self.fs*k/((nbin-1)*2)
            Cnn = np.sinc(2*fc*tau)                                     # spherical coherence matrix
            d, E = np.linalg.eigh(Cnn)
            d = np.maximum(d.real, 1e-3)
            iD = np.diag(1/np.sqrt(d))
            self.U[k,:,:] = np.dot(E, np.dot(iD, E.T.conj()))           # U = E*D^-0.5*E'     # ZCA whitening



    #----------------------------------------------------------------------------
    # whiten data using the whitening matrix Unn
    def whiten_data(self, Fz):

        nbin = Fz.shape[-2]
        if nbin != self.U.shape[0]:
            self.generate_whitening_matrix(nbin)

        # Fz.shape = (..., nbin, nmic)
        Fuz = np.einsum('kdc, ...kc->...kd', self.U, Fz)

        return Fuz



