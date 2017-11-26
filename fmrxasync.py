from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pyaudio
import Queue
import threading

def sdr_stream(samples, radio):
    global d
    d += 1
    print(d)
    if d > 60:
        sdr.cancel_read_async()
    else:
        dataQue.put(np.array(samples).astype("complex64"))


class fm_demod(threading.Thread):
    def run(self):
        while(1):
            x0 = dataQue.get()
            dataQue.task_done()

            x1 = x0 * np.exp(-1.0j*2.0*np.pi*fo/fs*np.arange(len(x0)))
            # check 1st 2000 samples, they seem bogus
            x2 = signal.lfilter(lpf, 1.0, x1)
            x3 = signal.decimate(x2, dec_rate)

            # demodulate
            adj_samp = x3[1:] * np.conj(x3[:-1])  # multiply adjacent samples, shorthand :)
            x4 = np.angle(adj_samp)
            x5 = signal.lfilter(dmf_tap1,dmf_tap2,x4)
            x6 = signal.decimate(x5, dec_audio)*8000
            x7 = x6.astype("int16")

            audioQue.put(x7)
            audioQue.join()


class play_radio(threading.Thread):
    def run(self):

        audioData = audioQue.get()
        audioQue.task_done();

        stream = p.open(format=pyaudio.paInt16,channels=1,rate=fa,output=True,frames_per_buffer=len(audioData)*2)

        while(1):
            stream.write(str(bytearray(audioData)))
            audioData = audioQue.get()
            audioQue.task_done()


fs = int(1.4e6) #225-300 kHz and 900 - 2032 kHz
fo = int(0.25e6) #250 kHz offset, to remove DC when sampling
fc = int(105.3e6) - fo
N = int(512*512) #multiple of 512
d = int(0)

# filter
lpf_taps = 64
lpf_bw = 0.2e6
lpf = signal.remez(lpf_taps, [0,lpf_bw,lpf_bw*2.6,fs/2], [1,0], Hz=fs)

# decimate
dec_rate = int(fs / lpf_bw)

# de-emphasis filter
dmf_shape = np.exp(-1/fs*dec_rate/75e-6)
dmf_tap1 = [1-dmf_shape]
dmf_tap2 = [1,-dmf_shape]

# convert to audio
fa = int(48e3)
dec_audio = int(fs/dec_rate/fa)
fs_audio = fs/dec_audio
p = pyaudio.PyAudio()

# init queues
dataQue = Queue.Queue([1])
audioQue = Queue.Queue([1])

fm_demod.daemon = True      # this lets the program exit when the thread is done
play_radio.daemon = True
fm_demod().start()
play_radio().start()

sdr = RtlSdr()
sdr.sample_rate = fs
sdr.center_freq = fc
sdr.gain = 'auto'

# begin reading from device
sdr.read_samples_async(sdr_stream, N)
sdr.close()
p.terminate()
