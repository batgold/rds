#!/usr/bin/env python
import numpy as nmp
import commpy.filters as com
import scipy.signal as sig
import constants
import graph

def memo(func):
    memo = {}
    def wrap(*args):
        if not args in memo:
            memo[args] = func()
        return memo[args]
    return wrap

def plot(func):
    memo = {}
    def wrap(*args):
        if not args in memo:
            memo[args] = func()
            #graph.response(memo[args], fs/rds_dec)
            graph.response(memo[args])
        return memo[args]
    return wrap

@memo
def rrc():
    """Root-Raised Cosine Filter"""
    n = int(121)
    T = 1/constants.fsym/2
    alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
    __, b = com.rrcosfilter(n, alfa, T, constants.fs/constants.rds_dec)
    return b, 1

@memo
def bpf_57k():
    n = 12
    cutoff = 3.0e3      # one-sided cutoff freq, slightly larger than 2.4kHz
    w = [(constants.fc - cutoff) / constants.fs*2,
            (constants.fc + cutoff) / constants.fs*2]
    b, a = sig.butter(N=n, Wn=w, btype='bandpass', analog=False)
    return b, a

@memo
def lpf():
    n = 8
    wn = 4*constants.fsym / (constants.fs/2)
    b, a = sig.butter(
            N=n, Wn=wn, btype='lowpass', analog=False)
    return b, a

@memo
def peak_19k():
    """Infinite (impulse response) Peak Filter at 19kHz"""
    w = constants.fpil/(constants.fs/2)
    #q = w / (500 / (constants.fs/2))        # Q = f/bw, BW = 500 Hz
    q = w / 16.0 * constants.fs
    b, a = sig.iirpeak(w, q)
    return b, a

@memo
def clk():
    """Infinite (impulse response) Peak Filter at Symbol Rate 1187.5Hz"""
    w = constants.fsym / float(constants.fs / constants.rds_dec / 2.0)
    q = w / 4.0 * constants.fs * constants.rds_dec         # Q = f/bw, BW = 4 Hz
    b, a = sig.iirpeak(w, q)
    return b, a

def build_costas_lpf(self):
    """COSTAS LOOP LPF"""
    N = self.costas_bw
    f = [0, 0.01, 0.02, 0.5]
    a = [1, 0]
    h = sig.remez(N, f, a)
    return h

def demph_eq():
    """De-Emphasis Filter"""
    cutoff = -1 / (constants.tau*constants.fs/constants.aud_dec)
    b = [1 - nmp.exp(cutoff)]
    a = [1, -nmp.exp(cutoff)]
    return b, a

def mono_lpf():
    n = 4
    cutoff = 17e3
    w = cutoff / fs * 2
    b, a = sig.butter(N=n, Wn=w, btype='lowpass', analog=False)
    return b, a

