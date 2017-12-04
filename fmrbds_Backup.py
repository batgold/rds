#!/usr/bin/env python

from rtlsdr import RtlSdr
import commpy as com
import matplotlib.pyplot as plt
import numpy as nmp
import scipy.signal as sig
sdr = RtlSdr()

f_sample = int(228e3)       # 225-300 kHz and 900 - 2032 kHz
#f_center = int(96.5e6)      # FM Station, need strong RDBS signal, SF
f_center = int(97.7e6)      # FM Station, need strong RDBS signal, PAL
f_pilot = int(19e3)         # pilot tone, 19kHz from Fc
n_samples = int(512*512*2)  # must be multiple of 512, should be a multiple of 16384 (URB size)
dec_rate = 24               # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
f_sym = 1187.5              # Symbol rate, full bi-phase
sps = f_sample/dec_rate/f_sym   # Samples per Symbol

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
    return pilot_b, pilot_a

def carrier_recovery(smp):
    pilot_b, pilot_a = pilot_filter()
    crr = sig.lfilter(pilot_b, pilot_a, smp)  # filter out 19 kHz
    crr3 = sig.hilbert(crr) ** 3.0             # multiply up to 57 kHz
    return crr, crr3

def build_filter_rds():         #BPF @ 57 kHz
    Wn = [(f_pilot*3 - 3e3)/f_sample*2, (f_pilot*3 + 3e3)/f_sample*2]   #slightly wider (3kHz)
    b,a = sig.butter(N=4, btype='bandpass', Wn=Wn, analog=False)
    #smp_rbds = sig.lfilter(b, a, smp)
    return b,a

def rrc_filter():
    # put 8 samples in a symbol period. i.e. 16+1 in the main lobe
    N = int(45)
    T = 1/1187.5/2
    alfa = 1
    t,rrc = com.filters.rrcosfilter(N,alfa,T,f_sample/dec_rate)
    plt.figure(2)
    plt.scatter(nmp.arange(N)-N/2,rrc)
    return rrc

def mix_and_decimate(smp, crr):
    smp_mix = smp * crr         # mix signal down by 57 kHz
    smp_bb = sig.decimate(smp_mix, dec_rate, zero_phase=True)  # decimate
    return smp_mix, smp_bb*1000

rrc = rrc_filter()
filter_rds = build_filter_rds()

smp_rtl = rtl_sample()                                  # RTL Samples
smp_demod = demodulate(smp_rtl)                         # FM Demodulate
crr,crr3 = carrier_recovery(smp_demod)                  # Pilot tone extraction
clk = (nmp.abs(crr) > 0)

smp_rbds = sig.lfilter(filter_rds[0], filter_rds[1], smp_demod)                 # Bandpass filter RBDS
smp_mix, smp_bb = mix_and_decimate(smp_rbds, crr3)      # Mix down to baseband
smp_bb2 = sig.convolve(rrc,smp_bb)

sym_bit = (smp_bb2 > 0)
phz = 1
symR = nmp.real(smp_bb2[phz::8])
symI = nmp.imag(smp_bb2[phz::8])

sym = (symR > 0)
bits = nmp.bitwise_xor(sym[1:], sym[:-1])

def rds_syndrome(n_bit):
    gx = 0x5B9 # 10110111001, g(x)=x^10+x^8+x^7+x^5+x^4+x^3+1
    syn = [383, 14, 303, 663, 748]
    name = ['A', 'B', 'C', 'D', 'C\'']
    reg = 0

    for n in range(26):
        reg = (reg<<1) | (bits[n_bit + n])
        if (reg & (1<<10)):
            reg = reg^gx
    for n in range(10,0,-1):
        reg = reg<<1
        if reg & (1<<10):
            reg = reg^gx

    word = reg & ((1<<10)-1)
    if syn.count(word) == 1:
        return name[syn.index(word)]
    return None

block_sync = []
prev_bit = 0
for n_bit in range(len(bits)-26):
    block_check = rds_syndrome(n_bit)
    if block_check:
        block_sync.append((n_bit-prev_bit, block_check))
        prev_bit = n_bit

print(block_sync)

# PLOTS
plt.figure(1)
plt.scatter(symR,symI)

f, (p1, p2, p3, p4) = plt.subplots(4)
p1.plot((smp_rbds[24000:26400]))
p2.plot(nmp.real(smp_mix[24000:26400]))
p3.plot(nmp.real(smp_bb2[1000:1100]))
p4.plot(nmp.real(sym_bit[1000:1100]))
#p4.plot(nmp.real(crr[1000:1100]))
plt.figure(2)

#plt.psd(smp_rbds, NFFT=2048, Fs=f_sample/1000, Fc=0)
#plt.psd(smp_bb2, NFFT=2048, Fs=f_sample/1000/dec_rate, Fc=0)
#plt.psd(pilot*100, NFFT=2048, Fs=f_sample/1000, Fc=0)
plt.show()
