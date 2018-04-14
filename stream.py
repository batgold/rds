#!/usr/bin/env python

import numpy as np
from rtlsdr import RtlSdr
from Queue import Queue
from threading import Thread
from graph import Graph

def callback(samples, sdr):
    d[0] += 1
    if d[0] > 5:
        sdr.cancel_read_async()
    else:
        #que.put(np.array(samples).astype("complex64"))
        que.put(samples)

def stream_rtlsdr(que):
    while True:
        x = que.get()
        que.task_done()
        #gph.fm(x)
        #print x[0:2]

que = Queue(0)

streamer = Thread(target=stream_rtlsdr, args=(que,))
streamer.daemon = True      # this lets the program exit when the thread is done
streamer.start()

F_SAMPLE = 228e3
N_SAMPLE = 512*512
d = [0]         #use this for recursive variable
gph = Graph(F_SAMPLE, N_SAMPLE)
sdr = RtlSdr()
sdr.sample_rate = F_SAMPLE
sdr.center_freq = 88.5e6
sdr.gain = 'auto'
sdr.read_samples_async(callback, N_SAMPLE)
sdr.close()
#gph.stop()


