#!/usr/bin/env python
from graph import Graph
from Queue import Queue
from filters import Filters
from threading import Thread
from pyaudio import PyAudio, paInt16
import numpy as nmp
import scipy.signal as sig
from constants import aud_dec

"""Stream FM Mono audio signal from RTL-SDR source."""

class Player():
    def __init__(self):
        self.que = Queue(0)
        self.player = PyAudio()
        self.stream = self.player.open(format=paInt16, channels=1, rate=Fa, output=True)

        audio_thread = Thread(target=self.play, args=(self.que,))
        audio_thread.daemon = True
        audio_thread.start()

def close(self):
    self.player.terminate
    self.stream.stop_stream()
    self.stream.close()

def demod_audio(x):
        self.graph = Graph(Fs, Ns)
        self.filters = Filters(Fs, 0, self.dec_rate)
    x3 = sig.lfilter(filters.mono_lpf[0], filters.mono_lpf[1], x2)
    x4 = sig.lfilter(filters.de_empf[0], filters.de_empf[1], x3)
    x5 = sig.decimate(x4, dec_rate, zero_phase=True)
    x6 = x5 * 5000
    x7 = x6.astype('int16')
    que.put(x7)

def play(self, que):
    while True:
        audio = que.get()
        que.task_done()
            self.stream.write(str(bytearray(audio)))
