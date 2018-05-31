#!/usr/bin/env/ python
import numpy as nmp
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
import matplotlib.patches as pch
from scipy.signal import welch
from constants import *
from Queue import Queue
from multiprocessing import Process
from PyQt4 import QtGui
import pyqtgraph as pyg

def display(name, q):
    print 'graph'

    while not q.empty():
        q.get()

    print q[0:2]
    app = QtGui.QApplication([])

    win = pyg.GraphicsWindow(title='BTG')
    p = win.addPlot()
    x = p.plot()

    app.exec_()

def memo(func):
    memo2 = {}
    def wrap(*args):
        if not args in memo2:
            memo2[args] = func(*args)
        return memo2[args]
    return wrap

@memo
class Graph:
    "start, put, update, join"

    def __init__(self):
        self.que = Queue()
        self.r = 0

    def _update(self):
        while not self.que.empty():
            print 'plot'
            data = self.que.get()
            self.line.setData(data)

    def run(self):
        white = pyg.mkColor("#E7E7F1"+"05")
        black = "#0E1019"

        app = QtGui.QApplication([])

        pyg.setConfigOption('background', black)
        win = pyg.GraphicsWindow(title='BTG')
        plot = win.addPlot()
        self.line = plot.plot(pen=white, alpha=0.1)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.showAxis('top')
        plot.showAxis('right')
        #plot.showLabel(axis='top',show=False)
        #plot.showLabel(axis='right',show=False)
        plot.setLabel(axis='bottom', text='SAMPLES')
        plot.setLabel(axis='left', text='AMP', units='V')

        axis = pyg.AxisItem('top',showValues=False)
        axis = pyg.AxisItem('top',showValues=True)
        #axis.showValues(False)

        print 'update'
        self._update()

        app.exec_()

class Graph2:

    def __init__(self, Fs, Ns, station):
        self.Fs = Fs
        self.Ns = Ns
        self.station = station / 1e6
        self.dec_rate = 12
        self.clk = []
        self.sym = []
        self.phz = []
        self.amp = []
        self.I = []
        self.Q = []

    def update(self, x2, x3, x4, phz, x6, clk, pi, pt, ps, rt, v_cnt, cnt, snr):
        self.fm = x2
        self.fm_bpf = x3
        self.bb = x4
        self.bb_rrc = x6
        self.phz_offset = phz
        self.clk = clk
        self.pi = pi
        self.pt = pt
        self.ps = ''.join(ps)
        self.rt = ''.join(rt)
        self.v_cnt = v_cnt
        self.cnt = cnt
        self.snr = nmp.around(snr, decimals=1)

        fig = plt.figure(1)
        plt.clf()
        #3x1 grid with narrow plot on bottom for text
        self.gs = grs.GridSpec(3,3,
                height_ratios=[4,4,1])

        self.scope()
        self.psd()
        self.constellation()
        self.text()
        self.costas()
        #plt.pause(3)
        plt.show()

        fig3 = plt.figure(2)
        self.scope2()
        plt.show()

    def psd(self):
        ax = plt.subplot(self.gs[0, :])
        N = 512

        f, Y = welch(self.fm, fs=self.Fs, nfft=N, return_onesided=False)
        f = f/1e3
        Y = 20 * nmp.log10(Y) + 130

        ax.set_xlim([0, max(f)])
        ax.set_ylim([0, 50])
        ax.set_xticks(nmp.arange(0, max(f), 10))
        ax.set_yticks(nmp.arange(0, 50, 5))
        ax.set_xlabel('FREQUENCY, KHz')
        ax.set_ylabel('POWER SPECTRAL DENSITY, dB/Hz')
        rds = pch.Rectangle((57-2.4, 0), 4.8, 100, alpha=0.4, facecolor='m')
        ax.add_patch(rds)
        plt.bar(f, Y, width=0.1)
        plt.text(57, 45, '{} dB'.format(self.snr), size=10, horizontalalignment='center')

    def scope(self):
        ax = plt.subplot(self.gs[1, 1])
        a = 60000
        b = a + 600
        t = nmp.arange(a, b)
        t2 = nmp.arange(a,b,6)
        y0 = self.fm_bpf[a-6:b-6]       #bpf causes 6 delay (12 taps)
        y1 = self.bb[a-6:b-6]
        y2 = self.bb_rrc[a/6 + 60:a/6+100 + 60] #RRC causes 60 delay (121 taps)
        y3 = self.clk[a/6 + 60:a/6+100 + 60] #RRC causes 60 delay (121 taps)
        plt.plot(t, y0)
        plt.plot(t, y1)
        plt.plot(t2, y2)
        plt.plot(t2, y3, 'm')

    def scope2(self):
        a = 50000
        b = a + 1200
        t = nmp.arange(a, b)
        t2 = nmp.arange(a,b,self.dec_rate)
        y0 = self.fm_bpf[a:b]       #bpf causes 6 delay (12 taps)
        y1 = self.bb[a:b]
        #y0 = self.fm_bpf[a-6:b-6]       #bpf causes 6 delay (12 taps)
        #y1 = self.bb[a-6:b-6]
        y2 = self.bb_rrc[a/self.dec_rate + 60:a/self.dec_rate+100 + 60] #RRC causes 60 delay (121 taps)
        y3 = self.clk[a/self.dec_rate + 60:a/self.dec_rate+100 + 60] #RRC causes 60 delay (121 taps)
        plt.plot(t, y0)
        plt.plot(t, y1)
        plt.plot(t2, y2)
        plt.plot(t2, y3, 'm')

    def constellation(self):
        ax = plt.subplot(self.gs[1, 2])
        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.set_xlabel(r'$\mathtt{I}$')
        ax.set_ylabel('Q')
        plt.plot(self.I, self.Q, '.', alpha=0.5)
        plt.plot([-1, 1], [0,0], 'C0')
        plt.plot([0,0], [-1, 1], 'C0')

    def costas(self):
        ax = plt.subplot(self.gs[1, 0])
        plt.plot(self.phz_offset)

    def text(self):
        ax = plt.subplot(self.gs[2, 0])
        plt.axis('off')
        plt.text(0, 0.8, '{}'.format(self.ps), size=20)
        plt.text(0, 0.2, '{}'.format(self.rt), size=20)
        ax = plt.subplot(self.gs[2, 1])
        plt.axis('off')
        plt.text(0.5, 0.8, '{}MHz - {} - {}'.format(self.station, self.pi,
            self.pt), size=20, horizontalalignment='center')
        ax = plt.subplot(self.gs[2, 2])
        plt.axis('off')
        plt.text(1, 0.8, '{} / {}'.format(self.v_cnt, self.cnt), size=20, \
                horizontalalignment='right')

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

    def bb2(self):
        N = 512
        f, Y = welch(self.bb, fs=self.Fs, nfft=N, return_onesided=False)
        f = f/1e3
        Y = 20 * nmp.log10(Y)
        plt.plot(f,Y)
        #plt.psd(self.bb, NFFT=2048, Fs=self.Fs/1000)
        #plt.psd(self.clk, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)

    def filter_response(self, b, a):
        w,h = sig.freqz(b,a)
        plt.plot(w/nmp.pi*self.Fs/2/1e3, abs(h))
        return None

def response(taps, f):
    b = taps[0]
    a = taps[1]
    w, h = sig.freqz(b,a)
    w = w/pi/2 * f/1e3
    #h = nmp.log10(h)
    h =abs(h)
    plt.figure()
    plt.plot(w,h)
    #plt.pause(1)
    #plt.show(block=False)
    plt.show()
