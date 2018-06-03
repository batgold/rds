#!/usr/bin/env python
#from Queue import Queue
#from threading import Thread
from pyaudio import PyAudio, paInt16
from constants import aud_dec, fa
import scipy.signal as sig

"""Stream FM Mono audio signal from RTL-SDR source."""
def receive(pipe_end, rpt):
    receive.rx_cnt = 0
    while True:
        msg = pipe_end.recv()
        receive.rx_cnt += 1
        if receive.rx_cnt == rpt:
            print 'end'
            break
        print 'got msg', receive.rx_cnt
        print max(msg)
        play(msg)

def play(x):
    x_dec = decimate(x)
    x_norm = normalize(x_dec)
    x_str = str(bytearray(x_norm.astype('int16')))
    player = PyAudio()
    stream = player.open(format=paInt16, channels=1, rate=int(fa), output=True)
    stream.write(x_str)

def decimate(x):
    return sig.decimate(x, aud_dec, zero_phase=True)

def normalize(x):
    """signal level of ~10k is good"""
    peak = max(x)
    x *= 1e4/peak
    return x

class Player():

    def __init__(self):
        #self.player = PyAudio()
        #self.stream = self.player.open(format=paInt16, channels=1, rate=int(fa), output=True)
        self.rx_cnt = 0

    #demph = filters.demph_eq()
    #x = sig.lfilter(demph[0], demph[1], x)
    def receive(self, pipe_end, rpt):
        self.receive.rx_cnt = 0
        while True:
            msg = pipe_end.recv()
            self.receive.rx_cnt += 1
            if self.receive.rx_cnt == rpt:
                #self.close()
                print 'end'
                break
            print 'got msg', self.receive.rx_cnt
            print max(msg)
            self.play(msg)

    def play(self, x):
        x_dec = self.decimate(x)
        x_norm = self.normalize(x_dec)
        x_str = str(bytearray(x_norm.astype('int16')))
        self.player = PyAudio()
        self.stream = self.player.open(format=paInt16, channels=1, rate=int(fa), output=True)
        self.stream.write(x_str)
        #self.stream.close()

    def close(self):
        self.player.terminate
        self.stream.stop_stream()
        self.stream.close()

    def decimate(self, x):
        return sig.decimate(x, aud_dec, zero_phase=True)

    def normalize(self, x):
        """signal level of ~10k is good"""
        peak = max(x)
        x *= 1e4/peak
        return x

class Player2():

    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.que = Queue(0)
        self.player = PyAudio()
        self.stream = self.player.open(format=paInt16, channels=1, rate=int(fa), output=True)
        self.start()

    def run(self):
        while True:
            audio = self.que.get()
            self.que.task_done()
            self.stream.write(str(bytearray(audio)))
            break

    def close(self):
        self.player.terminate
        self.stream.stop_stream()
        self.stream.close()
