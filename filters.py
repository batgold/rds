#!/usr/bin/env python

import numpy as nmp
import commpy as com
import scipy.signal as sig
import matplotlib.pyplot as plt

class Filters:

    def __init__(self, F_SAMPLE, F_SYMBOL, DEC_RATE):
        self.fs = F_SAMPLE
        self.fsym = F_SYMBOL
        self.dec = DEC_RATE

        if self.fsym == 0:
            self.mono_lpf = self.build_mono_lpf()
            self.de_empf = self.build_de_empf()
        else:
            self.fpilot = int(19e3)
            self.rrc = self.build_rrc()
            self.bpf = self.build_bpf()
            self.ipf = self.build_ipf()
            self.clk = self.build_clk()
            self.lpf = self.build_lpf()

    def build_rrc(self):
        """Cosine Filter"""
        N = int(121)
        T = 1/self.fsym/2
        alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
        __,rrc = com.filters.rrcosfilter(N, alfa, T, self.fs/self.dec)
        return rrc

    def build_bpf(self):
        """Bandpass Filter, at 57kHz"""
        cutoff = 3.0e3          # one-sided cutoff freq, slightly larger than 2.4kHz
        w = [(self.fpilot*3 - cutoff) / self.fs*2, (self.fpilot*3 + cutoff) / self.fs*2]
        b, a = sig.butter(N=12, Wn=w, btype='bandpass', analog=False)
        return b, a

    def build_ipf(self):
        """Infinite (impulse response) Peak Filter at 19kHz"""
        w = self.fpilot / float(self.fs / 2.0)
        q = w / 16.0 * self.fs        # Q = f/bw, BW = 16 Hz
        b, a = sig.iirpeak(w, q)
        return b, a

    def build_clk(self):
        """Infinite (impulse response) Peak Filter at Symbol Rate 1187.5Hz"""
        w = self.fsym / float(self.fs / self.dec / 2.0)
        q = w / 4.0 * self.fs * self.dec         # Q = f/bw, BW = 4 Hz
        b, a = sig.iirpeak(w, q)
        return b, a

    def build_lpf(self):
        w = self.fsym * 2 / self.fs * self.dec * 2
        b, a = sig.butter(N=9, Wn=w, btype='lowpass', analog=False)
        return b, a

    def build_mono_lpf(self):
        N = 4
        cutoff = 17e3
        w = cutoff / self.fs * 2
        return sig.butter(N=N, Wn=w, btype='lowpass', analog=False)

    def build_de_empf(self):
        """De-Emphasis Filter"""
        cutoff = self.fs * 75e-6 / self.dec
        #dmf_shape = nmp.exp(-1 / self.fs * self.dec / 75e-6)
        dmf_shape = nmp.exp(-1 / cutoff)
        return [1 - dmf_shape], [1, -dmf_shape]
