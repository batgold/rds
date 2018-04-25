#!/usr/bin/env python

import sys
import fmrx
import numpy as nmp
from rtlsdr import RtlSdr
from Queue import Queue
from threading import Thread

def callback(samples, sdr):
    d[0] += 1
    if d[0] > 5:
        sdr.cancel_read_async()
    else:
        #que.put(np.array(samples).astype("complex64"))
        que.put(samples)

def stream_rtlsdr(que):
    while True:
        samples = que.get()
        que.task_done()
        fmrx(samples)
        #print samples[0:2]

def main():
    d = [0]         #use this for recursive variable
    que = Queue(0)

    thread = Thread(target=stream_rtlsdr, args=(que,))
    thread.daemon = True      # this lets the program exit when the thread is done
    thread.start()

    STATION = float(sys.argv[1]) * 1e6
    F_SAMPLE = 228e3
    N_SAMPLE = 512*512

    sdr = RtlSdr()
    sdr.sample_rate = F_SAMPLE
    sdr.center_freq = STATION
    sdr.gain = 'auto'
    sdr.read_samples_async(callback, N_SAMPLE)
    sdr.close()

if __name__ == "__main__":
    main()
