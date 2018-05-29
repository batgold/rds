#!/usr/bin/env python
import numpy as nmp
#import commpy as com
import commpy.filters as com
import scipy.signal as sig
import graph
from constants import *
#from functools import wraps
import functools

#def memo(func):
#    memo = {}
#    def helper(*args):
#        if not args in memo:
#            memo[args] = func()
#        return memo[args]
#    return helper
#
@functools.lru_cache()
def demph_eq():
    """De-Emphasis Filter"""
    cutoff = -1 / (tau*fs/aud_dec)
    b = [1 - nmp.exp(cutoff)]
    a = [1, -nmp.exp(cutoff)]
    return b, a


def mono_lpf():
    n = 4
    cutoff = 17e3
    w = cutoff / fs * 2
    b, a = sig.butter(N=n, Wn=w, btype='lowpass', analog=False)
    return b, a

def build_rrc(self):
    """Cosine Filter"""
    N = int(121)
    T = 1/self.Fsym/2
    alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
    __,rrc = com.rrcosfilter(N, alfa, T, self.Fs/self.dec_rate)
    return rrc

def bpf():
    n = 12
    cutoff = 3.0e3          # one-sided cutoff freq, slightly larger than 2.4kHz
    w = [(fc - cutoff) / fs*2, (fc + cutoff) / fs*2]
    b, a = sig.butter(N=n, Wn=w, btype='bandpass', analog=False)
    return b, a

def build_bpf2(self):
    """Bandpass Filter, at 57kHz"""
    N = 2**8
    cutoff = 3.0e3          # one-sided cutoff freq, slightly larger than 2.4kHz
    bands = [0, self.Fc - cutoff-0.0001, self.Fc - cutoff, self.Fc + cutoff,
            self.Fc + cutoff + 0.0001, self.Fs/2]
    bands[:] = [b / self.Fs for b in bands]
    desired = [0, 1, 0]
    bpf = sig.remez(numtaps=N, bands=bands, desired=desired)
    self.graph.filter_response(bpf, 1)
    return bpf

def build_ipf(self):
    """Infinite (impulse response) Peak Filter at 19kHz"""
    w = self.fpilot / float(self.Fs / 2.0)
    q = w / 16.0 * self.Fs        # Q = f/bw, BW = 16 Hz
    b, a = sig.iirpeak(w, q)
    return b, a

def build_clk(self):
    """Infinite (impulse response) Peak Filter at Symbol Rate 1187.5Hz"""
    w = self.Fsym / float(self.Fs / self.dec_rate / 2.0)
    q = w / 4.0 * self.Fs * self.dec_rate         # Q = f/bw, BW = 4 Hz
    b, a = sig.iirpeak(w, q)
    #len 3
    return b, a

def build_lpf(self):
    w = self.Fsym * 2 / self.Fs * self.dec_rate * 2
    b, a = sig.butter(N=9, Wn=w, btype='lowpass', analog=False)
    return b, a


def build_costas_lpf(self):
    """COSTAS LOOP LPF"""
    N = self.costas_bw
    f = [0, 0.01, 0.02, 0.5]
    a = [1, 0]
    h = sig.remez(N, f, a)
    return h
