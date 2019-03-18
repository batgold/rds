#!/usr/bin/env/ python
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
import matplotlib.patches as pch
import constants as co

class Graph:

    def __init__(self):
        self.gs = grs.GridSpec(3,3, height_ratios=[4,4,1])

    def clf(self):
        plt.clf()

    def run(self):
        #plt.show()
        plt.pause(0.5)

    def scope(self, **kwargs):
        self.x_bpf = kwargs['x_bpf']
        self.bb_lpf = kwargs['bb_lpf']
        a = 6000
        b = a + 960*2
        delay = 40

        self.time = xrange(200, len(self.x_bpf)-200)
        self.x_bpf = self.x_bpf[200-delay:-200-delay]
        self.bb_lpf = self.bb_lpf[200:-200]

        ax = plt.subplot(self.gs[0, :])
        ax.plot(self.time, self.x_bpf, c='C1')
        ax.plot(self.time, self.bb_lpf, c='C0', lw=1.8)
        ax.set_xlim([a, b])
        ax.set_ylim([-0.2, 0.2])

    def spectrum(self, freq, x, snr):

        x_max = np.max(x) + 20

        # Rect((left,bottom), width, height)
        rds_patch = pch.Rectangle((57-2.4,-140), 4.8, 200, alpha=0.4, facecolor='m')

        ax = plt.subplot(self.gs[1, 1])
        ax.plot(freq*1e-3, x)
        ax.text(80, x_max-5, 'SNR=%.1fdB' % (snr,), fontsize=10)
        ax.set_ylim(bottom=-140, top=x_max)
        ax.add_patch(rds_patch)

    def eye(self, sym_list, sym_len, o):
        # only plot a few, it takes a long time to plot >1000
        ax = plt.subplot(self.gs[1,0])
        ax.set_ylim(-0.2, 0.2)
        for sym in sym_list[::10]:
            ax.plot(xrange(0,int(sym_len)), sym, c='C0', alpha=0.5)
        ax.plot([o,o],[-1,1], c='C2', lw=2)

    def constellation(self, i, q, rate):
        self.i = i
        self.q = q
        ax = plt.subplot(self.gs[1,2])
        ax.plot(self.i, self.q, ls='None', marker='.', alpha=0.5)

    def text(self, msg):
        pi, pt, gt, ps, rt, cnt = msg
        ax = plt.subplot(self.gs[2, 0])
        plt.axis('off')
        plt.text(0, 0.8, 'PS: {}'.format(ps), size=20)
        plt.text(0, 0.2, 'RT: {}'.format(rt), size=20)
        ax = plt.subplot(self.gs[2, 1])
        plt.axis('off')
        plt.text(0.5, 0.8, '{} - {}'.format(
            pi, pt), size=20, horizontalalignment='center')
        ax = plt.subplot(self.gs[2, 2])
        plt.axis('off')
        plt.text(1, 0.8, 'CNT: {}'.format(
            cnt), size=20, horizontalalignment='right')

def response(taps):
    b = taps[0]
    a = taps[1]
    w, h = sig.freqz(b,a)
    frq = w/np.pi/2 * constants.fs/1e3
    amp = 20*np.log10(abs(h))
    phz = np.unwrap(np.angle(h))*180/np.pi

    _, ax = plt.subplots()
    ax.plot(frq, amp, lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(frq, phz, c='C2', ls='--', lw=0.7)
    plt.show(block=False)

    plt.plot([0,0], [-1, 1], 'C0')
