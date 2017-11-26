#!/usr/bin/env python

from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as nmp
import scipy.signal as sig
sdr = RtlSdr()

f_sample = int(228e3)       # 225-300 kHz and 900 - 2032 kHz
f_center = int(96.5e6)      # FM Station, need strong RDBS signal
f_pilot = int(19e3)         # pilot tone, 19kHz from Fc
n_samples = int(512*512*2)  # must be multiple of 512, should be a multiple of 16384 (URB size)
dec_rate = 16               # RBDS rate 1187.5 Hz. Thus 288e3/1187.5/16 = sample/sec

def rtl_sample():
    sdr.sample_rate = f_sample
    sdr.center_freq = f_center
    sdr.gain = 'auto'
    smp_rtl = sdr.read_samples(n_samples)
    sdr.close()
    return smp_rtl[2000:]               #first 2000 samples are bogus

def demodulate(smp):
    smp_demod_cmplx = smp[1:] * nmp.conj(smp[:-1])  # 1:end, 0:end-1
    smp_demod = nmp.angle(smp_demod_cmplx)
    return smp_demod

def pilot_filter():
    pilot_wo = float(f_pilot)/float(f_sample)*2
    pilot_Q = pilot_wo / 5.0 * float(f_sample)  # Q = f/bw, BW = 5 Hz
    pilot_b, pilot_a = sig.iirpeak(pilot_wo, pilot_Q)

    print("Pilot Filter Q = %s" % str(pilot_Q))
    return pilot_b, pilot_a

def carrier_recovery(smp):
    pilot_b, pilot_a = pilot_filter()
    pilot = sig.lfilter(pilot_b, pilot_a, smp)  # filter out 19 kHz
    crr = sig.hilbert(pilot) ** 3.0             # multiply up to 57 kHz
    return crr

def mix_and_decimate(smp, crr, dec_rate):
    smp_mix = smp * crr         # mix signal down by 57 kHz
    smp_rbd = sig.decimate(smp_mix, dec_rate, zero_phase=True)  # decimate
    return smp_rbd

def data_shape_filter(smp_rbd):
    Wn = [55.8e3/f_sample*2, 58.2e3/f_sample*2]
    print(Wn)
    b,a = sig.butter(N=4, btype='bandpass', Wn=Wn, analog=False)
    return sig.lfilter(b, a, smp_rbd)

def data_shape_filter2(smp_rbd):
    Wn = 2.4e3/(f_sample/dec_rate)*2
    b,a = sig.butter(N=4, Wn=Wn, analog=False)
    smp_bb = sig.lfilter(b, a, smp_rbd)
    return smp_bb

def rms(x):
    y = x / nmp.sqrt(nmp.mean(x**2)) / nmp.sqrt(2)
    return y

smp_rtl = rtl_sample()
smp_demod = demodulate(smp_rtl)
crr = carrier_recovery(smp_demod)
#smp_rbd = mix_and_decimate(smp_demod, crr, dec_rate)
#smp_bb = data_shape_filter(smp_rbd)

smp_rbds = data_shape_filter(smp_demod)
smp_bb = mix_and_decimate(smp_rbds, crr, dec_rate)
#smp_test = rms(smp_test)

f, (p1, p2) = plt.subplots(2)
#subsmp = [v if i % 12 == 0 else 0 for i, v in enumerate(smp_bb_rms)]
#plt.plot(nmp.abs(subsmp[10500:11000]))
p1.plot((smp_rbds[10500:12000]))
p2.plot((smp_bb[10500:12000]))
#plt.plot((smp_bb_rms[10500:11000]))

#plt.psd(smp_demod, NFFT=2048, Fs=f_sample/1000, Fc=0)
#plt.psd(smp_test, NFFT=2048, Fs=f_sample/1000, Fc=0)
#plt.psd(smp_rbd, NFFT=2048, Fs=f_sample/1000/dec_rate, Fc=0)
#plt.plot(59.4*nmp.ones(10),nmp.arange(-100,0,10))
#plt.psd(pilot*100, NFFT=2048, Fs=f_sample/1000, Fc=0)
plt.show()
