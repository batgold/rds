#!/usr/bin/env python
import pyaudio
import constants
import scipy.signal as sig

def receive(pipe, n_max):
    receive.n = 0
    while True:
        data = pipe.recv()
        receive.n += 1
        if receive.n == n_max:
            break
        play(data)

def format(data):
    """Format FM Signal to int16 string with 10k amplitude"""
    decimated = sig.decimate(data, constants.aud_dec, zero_phase=True)
    normalized = 1e4*decimated/max(decimated)
    stringed = str(bytearray(normalized.astype('int16')))
    return stringed

def play(data):
    data = format(data)
    stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16, channels=1, rate=int(constants.fa), output=True)
    stream.write(data)
