#!/usr/bin/env python
import sys
import pickle
import demod
import rtlsdr
import multiprocessing as mpr
import player
import constants

def callback(samples, sdr):
    global n
    n += 1
    if n > n_max:
        q1.put(None)
        q2.put(None)
        sdr.cancel_read_async()
    else:
        fm = demod.demod_fm(samples)
        #parent.send(fm)
        q1.put(fm)
        print 'q1-put', n

        q2.put(fm)
        print 'q2-put', n

def read_file(filename):
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
    start_proc(samples)

def rx(q):
    while True:
        data = q.get(timeout=5)
        if data is None:
            print 'break'
            break
        print 'q1-get', data[0]
        player.play(data)

def rx2(q):
    while True:
        data = q.get(timeout=5)
        if data is None:
            print 'break'
            break
        print 'q2-get', data[0]


def read_rtlsdr(station):
    sdr = rtlsdr.RtlSdr()
    sdr.gain = 'auto'
    sdr.sample_rate = constants.fs
    sdr.center_freq = float(station) * 1e6
    sdr.read_samples_async(
            callback=callback, num_samples=constants.ns, context=sdr)
    sdr.close()

if __name__ == "__main__":
    n = 0
    n_max = 6

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    q1 = mpr.Queue()
    q2 = mpr.Queue()
    #parent, child = mpr.Pipe()

    #p1 = mpr.Process(target=read_rtlsdr, args=(arg2,))
    #p2 = mpr.Process(target=player.receive, args=(child, n_max))
    p1 = mpr.Process(target=rx, args=(q1,))
    p2 = mpr.Process(target=rx2, args=(q2,))

    p1.start()
    p2.start()

    read_rtlsdr(arg2)
    p1.join()
    p2.join()
    #if arg1 == '-f':
    #    read_file(arg1)
    #elif arg1 == '-r':
    #    read_rtlsdr(arg2)
