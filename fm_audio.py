from filters import Filters
from graph import Graph
from Queue import Queue
from threading import Thread
import numpy as nmp
import pyaudio
import scipy.signal as sig

class FM_Audio:
    """audio"""

    def __init__(self, F_SAMPLE, N_SAMPLE):
        self.F_AUDIO = int(44.1e3)
        self.audio_que = Queue(0)

        self.player = pyaudio.PyAudio()
        audio_thread = Thread(target=self.stream_audio,
                args=(self.audio_que,))
        audio_thread.daemon = True
        audio_thread.start()

    def close(self):
        self.player.terminate

    def fm_demod(self, samples, F_SAMPLE, N_SAMPLE):

        DEC_RATE = int(F_SAMPLE/self.F_AUDIO)

        graph = Graph(F_SAMPLE, N_SAMPLE)
        filters = Filters(F_SAMPLE, N_SAMPLE, DEC_RATE)

        x0 = samples[2000::]
        x1 = x0[1:] * nmp.conj(x0[:-1])    # 1:end, 0:end-1
        x2 = nmp.angle(x1)
        #graph.fm(x2)

        x3 = sig.lfilter(filters.mono_lpf[0], filters.mono_lpf[1], x2)
        x4 = x3
        #x4 = sig.decimate(x3, DEC_RATE, zero_phase=True)
        #graph.fm(x4*100)
        #graph.time(x3)
        x5 = sig.lfilter(filters.de_empf[0], filters.de_empf[1], x4)
        x6 = sig.decimate(x5, DEC_RATE, zero_phase=True)
        x7 = x6 * 5000
        x8 = x7.astype('int16')
        #graph.time(x8)
        print 'a'
        self.audio_que.put(x8)
        #self.audio_que.join()

    def stream_audio(self, audio_que):
        print 'b0'
        audio = audio_que.get()
        audio_que.task_done()
        audio_size = int(len(audio)*2)
        stream = self.player.open(format=pyaudio.paInt16, channels=1, rate=self.F_AUDIO, \
                output=True, frames_per_buffer=audio_size)

        while True:
            print 'b'
            stream.write(str(bytearray(audio)))
            audio = audio_que.get()
            audio_que.task_done()

        stream.stop_stream()
        stream.close()
