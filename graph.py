#!/usr/bin/env/ python
import numpy as nmp
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
import matplotlib.patches as pch
import scipy.signal as sig
from numpy.fft import rfft
from scipy.signal import welch

class Graph:

    def __init__(self, Fs, Ns, FM):
        self.Fs = Fs
        self.Ns = Ns
        self.FM = FM / 1e6
        self.fm = []
        self.fm_bpf = []
        self.clk = []
        self.bb = []
        self.bb2 = []
        self.bb3 = []
        self.sym = []
        self.phz_offset = []
        self.cos = []
        self.sin = []
        self.phz = []
        self.amp = []
        self.I = []
        self.Q = []
        self.pi = []
        self.pt = []
        self.ps = []
        self.rt = []

    def update(self):
        fig = plt.figure(1)
        plt.clf()
        #3x1 grid with narrow plot on bottom for text
        self.gs = grs.GridSpec(3,3,
                height_ratios=[4,4,1])

        self.scope()
        self.psd()
        self.constellation()
        #self.text()
        self.costas()
        #plt.pause(10)
        plt.show()

        fig2 = plt.figure(2)
        self.scope2()
        plt.show()

    def psd(self):
        ax = plt.subplot(self.gs[0, :])
        N = 512

        f, Y = welch(self.fm, fs=self.Fs, nfft=N, return_onesided=False)
        f = f/1e3
        Y = 20 * nmp.log10(Y) + 130

        fbb, Ybb = welch(self.fm_bpf, fs=self.Fs, nfft=N, return_onesided=False)
        fbb = fbb/1e3
        Ybb = 20 * nmp.log10(Ybb) + 150

        ax.set_xlim([0, max(f)])
        ax.set_ylim([0, 50])
        ax.set_xticks(nmp.arange(0, max(f), 10))
        ax.set_yticks(nmp.arange(0, 50, 5))
        ax.set_xlabel('FREQUENCY, KHz')
        ax.set_ylabel('POWER SPECTRAL DENSITY, dB/Hz')
        rds = pch.Rectangle((57-2.4, 0), 4.8, 100, alpha=0.4, facecolor='m')
        ax.add_patch(rds)
        plt.bar(f, Y, width=0.1)
        plt.bar(fbb, Ybb, width=0.1)
        #plt.plot([57, 57], [0, 50], 'C1', alpha=0.1, linewidth=100)

    def scope(self):
        ax = plt.subplot(self.gs[1, 1])
        a = 1000
        b = a + 300
        t = nmp.arange(a, b)
        y0 = self.fm_bpf[a:b]
        y1 = self.bb[a:b]
        y2 = self.bb2[a:b]
        plt.plot(t, y0)
        plt.plot(t, y1)
        plt.plot(t, y2)

    def scope2(self):
        a = 100000
        b = a + 5000
        t = nmp.arange(a, b)
        #t2 = nmp.arange(21000, 21000+1800)
        y0 = self.fm_bpf[a:b]
        y1 = self.bb[a:b]
        y2 = self.bb2[a:b]
        #y3 = self.bb3
        #plt.plot(t, y0)
        plt.plot(y1)
        plt.plot(y2)

    def constellation(self):
        ax = plt.subplot(self.gs[1, 2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$\mathtt{I}$')
        ax.set_ylabel('Q')
        plt.plot(self.I, self.Q, '.', alpha=0.5)
        plt.plot([-1, 1], [0,0], 'C0')
        plt.plot([0,0], [-1, 1], 'C0')

    def costas(self):
        ax = plt.subplot(self.gs[1, 0])
        plt.plot(self.phz_offset)

    def text(self):
        ax = plt.subplot(self.gs[2, :])
        plt.text(0, 1.0, 'TUNE: {} MHz'.format(self.FM))
        plt.text(0, 0.8, 'PI: {}'.format(self.pi))
        plt.text(0, 0.6, 'PT: {}'.format(self.pt))
        #plt.text(0, 0.4, 'PS: {}'.format(self.ps))
        plt.text(0, 0.2, 'RT: {}'.format(self.rt))
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

    def bb2(self, samples):
        plt.figure()
        plt.psd(samples, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        #plt.psd(self.clk, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        plt.ylim([-25, 0])
        plt.yticks(nmp.arange(-25, 0, 5))
        plt.show()

    def filter_response(self, b, a):
        w,h = sig.freqz(b,a)
        plt.plot(w/nmp.pi*self.Fs/2/1e3, abs(h))
        return None
