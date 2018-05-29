#!/usr/bin/env python
#from Queue import Queue
#from threading import Thread
from pyaudio import PyAudio, paInt16
from constants import aud_dec, fa

"""Stream FM Mono audio signal from RTL-SDR source."""

class Player():

    def __init__(self):
        self.player = PyAudio()
        self.stream = self.player.open(format=paInt16, channels=1, rate=int(fa), output=True)

    def play(self, audio):
        self.stream.write(str(bytearray(audio)))

    def close(self):
        self.player.terminate
        self.stream.stop_stream()
        self.stream.close()

#class Player2(Thread):
#
#    def __init__(self):
#        Thread.__init__(self)
#        self.daemon = True
#        self.que = Queue(0)
#        self.player = PyAudio()
#        self.stream = self.player.open(format=paInt16, channels=1, rate=int(fa), output=True)
#        self.start()
#
#    def run(self):
#        while True:
#            audio = self.que.get()
#            self.que.task_done()
#            self.stream.write(str(bytearray(audio)))
#            break
#
#    def close(self):
#        self.player.terminate
#        self.stream.stop_stream()
#        self.stream.close()
