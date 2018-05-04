#!/usr/bin/env/ python
import numpy as nmp
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
from scipy.fftpack import fft

class Graph:

    def __init__(self, Fs, Ns):
        self.Fs = Fs
        self.Ns = Ns
        self.fm = []
        self.I = []
        self.Q = []
        self.pi = []

    def update(self):
        plt.clf()
        #3x1 grid with narrow plot on bottom for text
        self.gs = grs.GridSpec(3,2,
                height_ratios=[4,4,1])

        self.scope()
        self.psd()
        self.constellation()
        self.text()
        plt.pause(1)

    def psd(self):
        ax = plt.subplot(self.gs[0, :])
        N = 512
        F = nmp.arange(0, N) * self.Fs/N/1e3/2
        Y = fft(self.fm, n=N*2)
        Y = nmp.abs(Y[:len(Y)/2])
        Y = 20 * nmp.log10(Y)
        ax.set_ylim([0, 50])
        ax.set_yticks(nmp.arange(0, 50, 10))
        ax.set_xticks(nmp.arange(0, max(F), 10))
        ax.set_xlabel('FREQUENCY, KHz')
        ax.set_ylabel('POWER SPECTRAL DENSITY, dB/Hz')
        plt.plot(F, Y)

    def scope(self):
        ax = plt.subplot(self.gs[1, 0])
        t = nmp.arange(0, 500)
        y = self.fm[:500]
        ax.set_ylabel('AMP')
        plt.plot(t, y)
        plt.plot(t*1.2, y*1.2)

    def constellation(self):
        ax = plt.subplot(self.gs[1, 1])
        #ax.set_ylim([-300, 300])
        #ax.set_xlim([-300, 300])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$\mathtt{I}$')
        ax.set_ylabel('Q')
        plt.plot(self.I, self.Q, '.', alpha=0.5)
        plt.plot([-300, 300], [0,0], 'w')
        plt.plot([0,0], [-300, 300], 'w')

    def text(self):
        ax = plt.subplot(self.gs[2, :])
        plt.text(0, 0.5, 'PI: {}'.format(self.pi))
        plt.axis('off')

    def time(self, samples):
        plt.figure()
        plt.plot(samples)
        plt.xlabel('SAMPLES')
        plt.ylabel('AMPLITUDE')
        plt.show()

    def time_domain(self, amp, phz, clk):
        plt.figure()
        plt.plot(amp, 'b.')
        plt.plot(phz, 'r+')
        plt.plot(clk, 'g')
        plt.show()

    def bb(self, samples):
        plt.figure()
        plt.psd(samples, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        #plt.psd(self.clk, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        plt.ylim([-25, 0])
        plt.yticks(nmp.arange(-25, 0, 5))
        plt.show()

    def filter_response(self):
        #f,h = sig.freqz(b,a)
        #plt.plot(f*self.fs/2/3.14,abs(h))
        #plt.show()
        return None
