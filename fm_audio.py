#!/usr/bin/env python
from graph import Graph
from Queue import Queue
from filters import Filters
from threading import Thread
from pyaudio import PyAudio, paInt16
import numpy as nmp
import scipy.signal as sig

class Audio:
    """Stream FM Mono audio signal from RTL-SDR source."""

    def __init__(self, Fs, Ns):
        Fa = int(44.1e3)
        self.dec_rate = int(Fs / Fa)
        self.que = Queue(0)
        self.player = PyAudio()
        self.stream = self.player.open(format=paInt16, channels=1, rate=Fa, output=True)
        self.graph = Graph(Fs, Ns)
        self.filters = Filters(Fs, 0, self.dec_rate)

        audio_thread = Thread(target=self.play, args=(self.que,))
        audio_thread.daemon = True
        audio_thread.start()

    def close(self):
        self.player.terminate
        self.stream.stop_stream()
        self.stream.close()

    def demod(self, samples):
        x0 = samples[2000:]
        x1 = x0[1:] * nmp.conj(x0[:-1])
        x2 = nmp.angle(x1)
        x3 = sig.lfilter(self.filters.mono_lpf[0], self.filters.mono_lpf[1], x2)
        x4 = sig.lfilter(self.filters.de_empf[0], self.filters.de_empf[1], x3)
        x5 = sig.decimate(x4, self.dec_rate, zero_phase=True)
        x6 = x5 * 5000
        x7 = x6.astype('int16')
        self.que.put(x7)

    def play(self, que):
        while True:
            audio = que.get()
            que.task_done()
            self.stream.write(str(bytearray(audio)))
