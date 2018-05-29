#!/usr/bin/env python
import sys
import pickle
from rtlsdr import RtlSdr
from demod import demod_fm
from constants import fs, ns

def callback(samples, sdr):
    global r
    r += 1
    if r > 3:
        sdr.cancel_read_async()
    else:
        demod_fm(samples, r)

def main():
    if source == '-f':
        filename = sys.argv[2]
        with open(filename, 'rb') as f:
            samples = pickle.load(f)
        demod_fm(samples, r)

    elif source == '-r':
        station = float(sys.argv[2]) * 1e6

        sdr = RtlSdr()
        sdr.sample_rate = fs
        sdr.center_freq = station
        sdr.gain = 'auto'
        sdr.read_samples_async(callback, ns)
        sdr.close()

if __name__ == "__main__":
    r = 0
    source = sys.argv[1]
    source_val = sys.argv[2]

    main()
