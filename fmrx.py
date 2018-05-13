#!/usr/bin/env python
import sys
import numpy as nmp
from rtlsdr import RtlSdr
from fm_rds import RDS
from fm_audio import Audio
from curses import wrapper
import curses

def callback(samples, sdr):
    global r
    r += 1
    if r > 1:
        sdr.cancel_read_async()
    else:
        #fm_audio.demod(samples)
        fm_rds.demod(samples)

def main():
    sdr = RtlSdr()
    sdr.sample_rate = Fs
    sdr.center_freq = station
    sdr.gain = 'auto'
    sdr.read_samples_async(callback, Ns)
    sdr.close()
    #audio.close()

if __name__ == "__main__":
    r = 0
    Fs = 228e3
    Ns = int(512*512)
    station = float(sys.argv[1]) * 1e6
    #fm_audio = Audio(Fs, Ns)
    fm_rds = RDS(0, Fs, Ns, station)

    main()
