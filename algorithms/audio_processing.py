# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import os
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fftpack import dct



#----------------------------------------------------------------
# read multichannel audio data
# output x.shape = (samples, nchan)
def audioread(filename, normalize=True):

    x, fs = sf.read(filename)

    if normalize==True:
        x = x*0.99/np.max(np.abs(x))

    return (x, fs)


#----------------------------------------------------------------
# write multichannel audio data
# input x.shape = (samples, nchan)
def audiowrite(x, filename, fs=16000, normalize=True):

    #x.shape = (samples, channels)

    if normalize==True:
        x = x*0.99/np.max(np.abs(x))

    mkdir(filename)
    sf.write(filename, x, fs)


#-------------------------------------------------------------------------
#convert STFT data back to time domain, and save to WAV-files
# data = tuple of STFT tensors
# filenames = tuple of file names
def convert_and_save_wavs(data, filenames, fs=16000):

    for Fz, filename in zip(data, filenames):
        z = mistft(Fz)                                  # Fz.shape = (nfram, self.nbin)

        mkdir(os.path.dirname(filename))
        audiowrite(z, filename, fs=fs)



#----------------------------------------------------------------
# wrapper for python real fft
def rfft(Bx, n=None, axis=-1):

    Fx = np.fft.rfft(Bx, n=n, axis=axis).astype(np.complex64)

    return Fx


#----------------------------------------------------------------
# wrapper for python real ifft
def irfft(Fx, n=None, axis=-1):

    Bx = np.fft.irfft(Fx, n=None, axis=axis).astype(np.float32)

    if n is not None:
        samples = Bx.shape[axis]
        Bx = np.moveaxis(Bx, axis, 0)
        if n<samples:
            Bx = Bx[:n,...]
        else:
            shape = (n,) + Bx.shape[1:]
            By = np.zeros(shape=shape, dtype=Bx.dtype)
            By[:samples,...] = Bx
        Bx = np.moveaxis(By, 0, axis)

    return Bx



#----------------------------------------------------------------
# perform a multichannel STFT on audio data x, along <axis>
# <axis> of shape (samples,) is replaced with two new dimensions: (nfram, nbin)
def mstft(x, wlen=1024, shift=256, window=signal.blackman, axis=-1):

    # move <axis> to last dimension
    x = np.swapaxes(x, axis, -1)
    x = np.asarray(x, dtype=np.float32)                          # shape = (..., samples)

    x_shape = tuple(x.shape[:-1])
    samples = x.shape[-1]

    nbin = int(wlen/2+1)
    nfram = int(np.ceil( (samples-wlen+shift)/shift ))
    samples_padding = nfram*shift+wlen-shift - samples

    pad = np.zeros(x_shape+(samples_padding,), dtype=np.float32)
    x = np.concatenate([x, pad], axis=-1)

    Bx = np.zeros(x_shape+(nfram, wlen), dtype=np.float32)
    idx = np.arange(wlen)
    for t in range(nfram):
        Bx[...,t,:] = x[...,idx+t*shift]

    analysis_window = window(wlen)
    Bx = np.einsum('...tw,w->...tw', Bx, analysis_window)
    Fx = rfft(Bx, n=wlen, axis=-1).astype(np.complex64)          # shape = (..., nfram, nbin)

    # restore original dimensions
    if x.ndim>1:
        Fx = np.swapaxes(Fx, axis, -1)
        Fx = np.swapaxes(Fx, axis, -2)

    return Fx



#----------------------------------------------------------------
# perform a multichannel inverse STFT on audio data Fx
# Fx.shape = (nbin, nfram, nchan)
# output x.shape = (samples, nchan)
def mistft(Fx, wlen=1024, shift=256, window=signal.blackman):

    assert (Fx.ndim == 2 or Fx.ndim == 3), 'Fx must have either 2 or 3 dimensions'

    Fx = np.asarray(Fx, dtype=np.complex64)
    nbin = Fx.shape[0]
    nfram = Fx.shape[1]
    samples = nfram*shift+wlen-shift

    analysis_window = window(wlen)

    assert np.mod(wlen, shift) == 0
    number_of_shifts = int(wlen/shift)

    sum_of_squares = np.zeros(shift)
    for i in range(number_of_shifts):
        idx = np.arange(shift) + i*shift
        sum_of_squares = sum_of_squares + np.abs(analysis_window[idx])**2

    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares

    if Fx.ndim == 2:
        x = np.zeros((samples,), dtype=np.float32)
        for t in range(nfram):
            Bx = np.real(np.fft.irfft(Fx[:,t]))
            idx = np.arange(wlen) + t*shift
            x[idx] += Bx * synthesis_window


    if Fx.ndim == 3:
        nchan = Fx.shape[2]
        x = np.zeros((samples, nchan), dtype=np.float32)
        for c in range(nchan):
            for t in range(nfram):
                Bx = np.real(np.fft.irfft(Fx[:,t,c]))
                idx = np.arange(wlen) + t*shift
                x[idx,c] += Bx * synthesis_window


    return x



#------------------------------------------------------------------------------
# get amplitude response of a highpass filter with <order> 
# fs = samplerate
# fc = corner frequency
# response at H(fc) = 1/sqrt(2)
# nbin = number of frequency bins
def get_highpass_filter(nbin=513, fs=16e3, fc=100, order=2):

    fvect = np.arange(nbin)*fs/(2*(nbin-1))
    k = np.power( np.sqrt(2)-1 , -1/order)
    tmp = np.power( (fvect/fc)*k , order )
    H = np.maximum( 1-1/(1+tmp) , 1e-6 )

    return H



#------------------------------------------------------------------------------
def apply_highpass_filter(Fx, fs=16e3, fc=100, order=2):

    nbin = Fx.shape[0]

    H = get_highpass_filter(nbin=nbin, fs=fs, fc=fc, order=order)
    Fz = np.zeros_like(Fx)
    for k in range(nbin):
        Fz[k,...] = Fx[k,...]*H[k]

    return Fz



#------------------------------------------------------------------------------
def hz_to_mel(hz):

    return 2595*np.log10(1+hz/700)



#------------------------------------------------------------------------------
def mel_to_hz(mel):

    return 700*(10**(mel/2595)-1)



#------------------------------------------------------------------------------
def create_mel_filterbank(nbin=513, fs=16e3, nband=40):

    low_freq_mel = hz_to_mel(100)
    high_freq_mel = hz_to_mel(fs/2)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nband+2, dtype=np.float32)        # equally spaced mel scale with <nband> kernels
    hz_points = mel_to_hz(mel_points)

    bin_index = np.asarray(np.floor(2*nbin*hz_points/fs), dtype=np.int32)
    filterbank = np.zeros((nband, nbin), dtype=np.float32)

    for m in range(1, nband+1):

        # create <nband> triangular kernels
        f_m_left = bin_index[m-1]
        f_m_center = bin_index[m]
        f_m_right = bin_index[m+1]

        for k in range(f_m_left, f_m_center):
            filterbank[m-1, k] = (k-f_m_left) / (f_m_center-f_m_left)

        for k in range(f_m_center, f_m_right):
            filterbank[m-1, k] = (f_m_right-k) / (f_m_right-f_m_center)

    return filterbank                  # shape = (nband, nbin)



#------------------------------------------------------------------------------
def convert_to_mel(Fx, filterbank):

    Px = np.abs(Fx)**2
    Mx = np.dot(Px, filterbank.T)
    mel = np.log(Mx + 1e-3)

    return mel



#------------------------------------------------------------------------------
def convert_to_mfcc(Fx, filterbank):

    Px = np.abs(Fx)**2
    Mx = np.dot(Px, filterbank.T)
    Mx = np.log(Mx + 1e-3)

    mfcc = dct(Mx, axis=-1, type=2, norm='ortho')

    return mfcc



#------------------------------------------------------------------------------
def mkdir(path):

    if not os.path.exists(os.path.dirname(path)): 
        os.makedirs(os.path.dirname(path))



#-------------------------------------------------------------------------
def si_sdr(s, y, average=True):

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

    if average is True:
        return np.mean(si_sdr)
    else:
        return si_sdr



