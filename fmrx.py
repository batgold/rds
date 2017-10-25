from rtlsdr import RtlSdr
import numpy as np
import matplotlib
import scipy.signal as signal
import matplotlib.pyplot as plt
import pyaudio

sdr = RtlSdr()

fs = int(2e6) #225-300 kHz and 900 - 2032 kHz
N = int(512*16000)
# N = int(256*1024*16)
fc = int(88.5e6)

sdr.sample_rate = fs
sdr.center_freq = fc
sdr.gain = 'auto'
stream = sdr.read_samples(N)
print(sdr.get_gains())
print(sdr.get_gain())
sdr.close()
del(sdr)

x1 = np.array(stream[range(2000,N-2000-1)]) #first 2000 samples are bogus


# filter
lpf_taps = 64
lpf_bw = 0.2e6
lpf = signal.remez(lpf_taps, [0,lpf_bw,lpf_bw*3,fs/2], [1,0], Hz=fs)

x2 = signal.lfilter(lpf, 1.0, x1)



# decimate
dec_rate = int(fs / lpf_bw)
x3 = x2[0::dec_rate]

print("Rate = " + str(fs/dec_rate/1000) + "kHz")



# demodulate
adj_samp = x3[1:] * np.conj(x3[:-1])  # multiply adjacent samples, shorthand :)
x4 = np.angle(adj_samp)


# de-emphasis filter
dmf_shape = np.exp(-1/fs*dec_rate/75e-6)
dmf_tap1 = [1-dmf_shape]
dmf_tap2 = [1,-dmf_shape]
x5 = signal.lfilter(dmf_tap1,dmf_tap2,x4)



# convert to audio
fa = int(44.1e3)
dec_audio = int(fs/dec_rate/fa)
fs_audio = fs/dec_audio

x6 = signal.decimate(x5, dec_audio)*4000
x7 = x6.astype("int16")

plt.plot(range(0,len(x6)),x6,color="blue")
plt.plot(range(0,len(x7)),x7,color="red")
# plt.show()

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,channels=1,rate=fa,output=True,frames_per_buffer=int(len(x7)*2))


while(1):
    stream.write(str(bytearray(x7)))

stream.stop_stream()
stream.close()

p.terminate
