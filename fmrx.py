#!/usr/bin/env python

import sys
#import fm_rds
import numpy as nmp
from rtlsdr import RtlSdr
from Queue import Queue
from threading import Thread
from fm_audio import fm

def callback(samples, sdr):
    global r
    r += 1
    if r > 5:
        sdr.cancel_read_async()
    else:
        fm(samples, F_SAMPLE, N_SAMPLE)
        #que.put(samples)

def stream_rtlsdr(que):
    while True:
        samples = que.get()
        que.task_done()
        #fm(samples, N_SAMPLE, F_SAMPLE)

def main():

    thread = Thread(target=stream_rtlsdr, args=(que,))
    thread.daemon = True      # this lets the program exit when the thread is done
    thread.start()

    sdr = RtlSdr()
    sdr.sample_rate = F_SAMPLE
    sdr.center_freq = STATION
    sdr.gain = 'auto'
    sdr.read_samples_async(callback, N_SAMPLE)
    sdr.close()

if __name__ == "__main__":
    r = 0
    que = Queue(0)

    F_SAMPLE = 228e3
    N_SAMPLE = 512*10
    STATION = float(sys.argv[1]) * 1e6

    main()
