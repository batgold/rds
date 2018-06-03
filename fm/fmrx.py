#!/usr/bin/env python
import sys
import pickle
import demod
import graph
from rtlsdr import RtlSdr
from constants import fs, ns
from Queue import Queue
import multiprocessing as mpr
from threading import Thread
import numpy as nmp
import player

def start_proc(samples):
    fm = demod.demod_fm(samples)
    #gr = graph.Graph()
    #rds_proc = mpr.Process(target=demod.demod_rds, args=(fm,))
    parent_conn, child_conn = multiprocessing.Pipe()

    audio_proc = mpr.Process(target=demod.demod_audio, args=(parent_conn, fm))
    graph_proc = mpr.Process(target=gr.run, args=(child_conn,))

    #audio_thread = Thread(target=demod.demod_audio, args=(fm,))
    #audio_thread.start()

    #rds_thread = Thread(target=demod.demod_rds, args=(fm,))
    audio_proc.start()

def callback(samples, sdr):
    global r
    r += 1
    if r > rpt:
        sdr.cancel_read_async()
    else:
        print 'sent', r
        fm = demod.demod_fm(samples)
        parent.send(fm)

def audio(child):
    audio.count = 0
    while True:
        msg = child.recv()
        audio.count += 1
        if audio.count == rpt:
            break
        player.Player().play(msg)

def read_file(filename):
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
    start_proc(samples)

def read_rtlsdr(parent, station):
    station = float(station) * 1e6
    sdr = RtlSdr()
    sdr.sample_rate = fs
    sdr.center_freq = station
    sdr.gain = 'auto'
    sdr.read_samples_async(
            callback=callback, num_samples=ns, context=sdr)
    sdr.close()

if __name__ == "__main__":
    r = 0
    rpt = 8
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    parent, child = mpr.Pipe()

    p1 = mpr.Process(target=read_rtlsdr, args=(parent, arg2))
    #p2 = mpr.Process(target=audio, args=(child,))
    p2 = mpr.Process(target=player.receive, args=(child, rpt))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    #if arg1 == '-f':
    #    read_file(arg1)
    #elif arg1 == '-r':
    #    read_rtlsdr(arg2)
