#!/usr/bin/env python

import numpy as nmp
import commpy as com
import scipy.signal as sig
import matplotlib.pyplot as plt
from graph import Graph

class Filters:

    def __init__(self, Fs, Fsym, dec_rate):
        self.Fs = Fs
        self.Fc = 57e3
        self.Fsym = Fsym
        self.dec_rate = dec_rate
        self.graph = Graph(Fs, 1, 1)
        self.costas_bw = 2**4

        if self.Fsym == 0:
            self.mono_lpf = self.build_mono_lpf()
            self.de_empf = self.build_de_empf()
        else:
            self.fpilot = int(19e3)
            self.rrc = self.build_rrc()
            self.bpf = self.build_bpf()
            #self.build_bpf2()
            #self.ipf = self.build_ipf()
            self.clk = self.build_clk()
            self.lpf = self.build_lpf()
            self.costas_lpf = self.build_costas_lpf()

    def build_rrc(self):
        """Cosine Filter"""
        N = int(121)
        T = 1/self.Fsym/2
        alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
        __,rrc = com.filters.rrcosfilter(N, alfa, T, self.Fs/self.dec_rate)
        return rrc

    def build_bpf(self):
        N = 12
        cutoff = 3.0e3          # one-sided cutoff freq, slightly larger than 2.4kHz
        w = [(self.Fc - cutoff) / self.Fs*2, (self.Fc + cutoff) / self.Fs*2]
        b, a = sig.butter(N=N, Wn=w, btype='bandpass', analog=False)
        #self.graph.filter_response(b, a)
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
        return b, a

    def build_lpf(self):
        w = self.Fsym * 2 / self.Fs * self.dec_rate * 2
        b, a = sig.butter(N=9, Wn=w, btype='lowpass', analog=False)
        return b, a

    def build_mono_lpf(self):
        N = 4
        cutoff = 17e3
        w = cutoff / self.Fs * 2
        return sig.butter(N=N, Wn=w, btype='lowpass', analog=False)

    def build_de_empf(self):
        """De-Emphasis Filter"""
        cutoff = self.Fs * 75e-6 / self.dec_rate
        #dmf_shape = nmp.exp(-1 / self.Fs * self.dec_rate / 75e-6)
        dmf_shape = nmp.exp(-1 / cutoff)
        return [1 - dmf_shape], [1, -dmf_shape]

    def build_costas_lpf(self):
        """COSTAS LOOP LPF"""
        N = self.costas_bw
        f = [0, 0.01, 0.02, 0.5]
        a = [1, 0]
        h = sig.remez(N, f, a)
        return h
