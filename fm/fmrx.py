#!/usr/bin/env python
import sys
import pickle
import demod
from rtlsdr import RtlSdr
from constants import fs, ns
from Queue import Queue
from threading import Thread

def start_proc(samples):
    fm = demod.demod_fm(samples)
    audio_thread = Thread(target=demod.demod_audio, args=(fm))
    rds_thread = Thread(target=demod.demod_rds, args=(fm, q))
    audio_thread.start()
    rds_thread.start()

def callback(samples, sdr):
    global r
    r += 1
    if r > 3:
        sdr.cancel_read_async()
    else:
        start_proc(samples)

def main():

    if source == '-f':
        filename = sys.argv[2]
        with open(filename, 'rb') as f:
            samples = pickle.load(f)
        start_proc(samples)

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
    q = Queue()
    source = sys.argv[1]
    source_val = sys.argv[2]

    main()