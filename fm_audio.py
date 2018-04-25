from filters import Filters
from graph import Graph
import matplotlib.pyplot as plt
import numpy as nmp
import pyaudio
import scipy.signal as sig

def main(a, b, c):
    #samples = float(sys.argv[1])
    #N_SAMPLE = float(sys.argv[2])
    #F_SAMPLE = float(sys.argv[3])
    print b, c

def fm(samples, F_SAMPLE, N_SAMPLE):

    x1 = samples[2000::]
    x1 = samples[1:] * nmp.conj(samples[:-1])    # 1:end, 0:end-1
    x2 = nmp.angle(x1)
    #graph.fm(x2)
    #self.calc_snr(x2)
    #crr = self.recover_carrier(x2)  # get 57kHz carrier
    x3 = sig.lfilter(filters.bpf[0], filters.bpf[1], x2) # BPF
    x4 = 1000 * crr * x3        # mix down to baseband
    x5 = sig.decimate(x4, DEC_RATE, zero_phase=True)
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


if __name__ == "__main__":
    main()
