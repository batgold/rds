#!/usr/bin/env python
import pyaudio
import mute_alsa
import constants as co
import scipy.signal as sg

def receive(que):
    """Get Data From Demod"""
    while True:
        data = que.get(timeout=5)
        if data is None:
            break
        play(data)

def format(x):
    """Format FM Signal to int16 string with 10k amplitude"""
    x_dec = sg.decimate(x, co.aud_dec, zero_phase=True)
    x_norm = 1e4 * x_dec/max(x_dec)
    x_str = str(bytearray(x_norm.astype('int16')))
    return x_str

def play(data):
    """Play FM Audio"""
    data = format(data)
    stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=int(co.fa),
            output=True)
    stream.write(data)
