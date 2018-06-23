#!/usr/bin/env/ python
import numpy as nmp
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
import matplotlib.patches as pch
import scipy.signal
import constants
import filters

def memo(func):
    memo2 = {}
    def wrap(*args):
        if not args in memo2:
            memo2[args] = func(*args)
        return memo2[args]
    return wrap

class Graph:

    def scope(self, x_bpf, bb_lpf, bb_dec, bb_phz, bb_i):
        #natural rate
        a = 48000
        b = a + 2*960
        #delay = 40
        delay = 60
        self.time = nmp.arange(a, b)
        self.x_bpf = x_bpf[a-delay:b-delay]
        self.bb_lpf = bb_lpf[a:b]

        #decimated rate
        a = a/constants.rds_dec
        b = a + 2*960/constants.rds_dec
        self.time_dec = nmp.arange(
                a*constants.rds_dec, b*constants.rds_dec, constants.rds_dec)
        self.bb_dec = bb_dec[a:b]
        self.bb_phz = 0.2/nmp.pi*bb_phz[a:b]
        self.bb_i = bb_i[a:b]

    def spectrum(self, x):
        n = 512
        freq, power = scipy.signal.welch(
                x, fs=constants.fs, nfft=n, return_onesided=True)
        self.freq = freq * 1e-3
        self.power = 20 * nmp.log10(power)

    def spectrum2(self, x):
        n = 512
        freq, power = scipy.signal.welch(
                x, fs=constants.fs/constants.rds_dec, nfft=n, return_onesided=True)
        self.freq2 = freq * 1e-3
        self.power2 = 20 * nmp.log10(power)

    def constellation(self, i, q, rate):
        self.i = i
        self.q = q

    def run(self):
        gs = grs.GridSpec(3,3, height_ratios=[4,4,1])

        #scope
        ax = plt.subplot(gs[0, :])
        ax.plot(self.time, self.x_bpf, c='C1')
        ax.plot(self.time, self.bb_lpf, c='C0', lw=1.8)
        ax.plot(self.time_dec, self.bb_dec, c='C2', ls='None', marker='D')
        #ax.plot(self.time_dec, self.bb_phz, c='r')
        ax.plot(self.time_dec, self.bb_i, c='m', lw=4)
        ax.set_ylim([-0.2, 0.2])

        #spectrum
        ax = plt.subplot(gs[1, 1])
        ax.plot(self.freq, self.power)
        ax.set_ylim(bottom=-140)

        #spectrum
        ax = plt.subplot(gs[1, 0])
        ax.plot(self.freq2, self.power2)

        #constellation
        ax = plt.subplot(gs[1,2])
        ax.plot(self.i, self.q, ls='None', marker='.', alpha=0.5)

        plt.pause(0.1)





@memo
class Graph3:
    "start, put, update, join"

    def __init__(self):
        self.que = Queue()
        #self.make_graph()
        self.r = 0

    def _update(self):
        while not self.que.empty():
            print 'plot'
            data = self.que.get()
            #self.line.setData(data)

    def make_graph(self):
        white = pyg.mkColor("#E7E7F1"+"05")
        black = "#0E1019"

        self.app = QtGui.QApplication([])

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
        self.app.exec_()

    def run(self):
        print 'update'
        self._update()

class Graph2:

    def init(self):
        self.clk = []
        self.sym = []
        self.phz = []
        self.amp = []
        self.I = []
        self.Q = []

    def update(self, x2, x3, x4, phz, x6, clk, pi, pt, ps, rt, v_cnt, cnt, snr):
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

def response(taps):
    b = taps[0]
    a = taps[1]
    w, h = sig.freqz(b,a)
    frq = w/nmp.pi/2 * constants.fs/1e3
    amp = 20*nmp.log10(abs(h))
    phz = nmp.unwrap(nmp.angle(h))*180/nmp.pi

    _, ax = plt.subplots()
    ax.plot(frq, amp, lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(frq, phz, c='C2', ls='--', lw=0.7)
    plt.show(block=False)
