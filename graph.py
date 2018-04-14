#!/usr/bin/env/ python

import numpy as nmp
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

class Graph:

    def __init__(self, fs, n):
        self.F_SAMPLE = fs
        self.N_SAMPLE = n

        self.x = nmp.arange(0, self.N_SAMPLE)
        self.y = nmp.zeros(self.N_SAMPLE)

        #plt.ion()
        #plt.figure()
        #self.fig = plt.figure()
        #ax = plt.figure.add_subplot(111)
        #self.spec, = ax.plot(self.x, self.y)
        #self.psd = ax.psd(nmp.arange(0,2e4), NFFT=2048, Fs=self.F_SAMPLE/1000)


        #plt.ion()
        #self.fig, self.ax = plt.subplots()
        #self.ax.plot(self.x,self.y)

    def fm_bad(self, samples):
        self.spec.set_ydata(samples)
        plt.figure.canvas.draw()

        #self.ax.plot(self.x, samples)
        #self.fig.canvas.draw_idle()
        #time.sleep(0.05)
        return None

    def fm(self, samples):
        #plt.figure()
        plt.ylim([-50, 0])
        plt.yticks(nmp.arange(-50, 0, 10))
        plt.psd(samples, NFFT=2048, Fs=self.F_SAMPLE/1000)
        #plt.draw()
        plt.pause(0.005)

    def stop(self):
        #plt.close()
        #plt.ioff()
        #plt.show()
        return None

    def constellation(self, I, Q):
        plt.figure()
        plt.plot(I, Q, 'b.', alpha=0.5)
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

