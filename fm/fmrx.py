#!/usr/bin/env python
import sys
import pickle
import demod
import rtlsdr
import player
import constants
import multiprocessing

max_calls = 9
@rtlsdr.limit_calls(max_calls)
def sdr_callback(samples, sdr):
    fm = demod.demod_fm(samples)
    aud_que.put(fm)
    rds_que.put(fm)

def read_file(filename):
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
    fm = demod.demod_fm(samples)
    aud_que.put(fm)
    rds_que.put(fm)

def read_rtlsdr(station):
    sdr = rtlsdr.RtlSdr()
    sdr.gain = 400
    sdr.sample_rate = constants.fs
    sdr.center_freq = float(station) * 1e6
    sdr.read_samples_async(
            callback=sdr_callback, num_samples=constants.ns, context=sdr)
    sdr.close()

if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    aud_que = multiprocessing.Queue()
    rds_que = multiprocessing.Queue()
    aud_proc = multiprocessing.Process(target=player.receive, args=(aud_que,))
    rds_proc = multiprocessing.Process(target=demod.receive, args=(rds_que,))

    aud_proc.start()
    rds_proc.start()

    if arg1 == '-f':
        read_file(arg2)

    elif arg1 == '-r':
        read_rtlsdr(arg2)

    aud_que.put(None)
    rds_que.put(None)

    aud_proc.join()
    rds_proc.join()

#geeksforgeeks.com/multiprocessing
