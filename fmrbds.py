#!/usr/bin/env python

import pickle
import commpy as com
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig
from rtlsdr import RtlSdr


F_SAMPLE = int(228e3)       # 225-300 kHz and 900 - 2032 kHz
F_CENTER = int(96.5e6)      # FM Station, need strong RDBS signal, SF
#F_CENTER = int(97.7e6)      # FM Station, need strong RDBS signal, PAL
F_PILOT = int(19e3)         # pilot tone, 19kHz from Fc
N_SAMPLES = int(512*512*2)  # must be multiple of 512, should be a multiple of 16384 (URB size)
DEC_RATE = 12               # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
F_SYM = 1187.5              # Symbol rate, full bi-phase
sps = int(F_SAMPLE/DEC_RATE/F_SYM)   # Samples per Symbol

print("Samples/sec", str(sps))


def rtl_sample():
    """STEP 1: SAMPLE DATA FROM RTLSDR"""
    sdr.sample_rate = F_SAMPLE
    sdr.center_freq = F_CENTER
    sdr.gain = 'auto'
    smp_rtl = sdr.read_samples(N_SAMPLES)
    sdr.close()
    return smp_rtl[2000:]               #first 2000 samples are bogus


class Filter:
    """Construct Filters"""

    def rrc(self):
        """Cosine Filter"""
        N = int(120)
        T = 1/1187.5*3/8
        alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
        __,rrc = com.filters.rrcosfilter(N,alfa,T,F_SAMPLE/DEC_RATE)
        return rrc


    def bpf(self):
        """Bandpass Filter, at 57kHz"""
        cutoff = 3.0e3          # one-sided cutoff freq, slightly larger than 2.4kHz
        w = [(F_PILOT*3 - cutoff) / F_SAMPLE*2, (F_PILOT*3 + cutoff) / F_SAMPLE*2]
        b, a = sig.butter(N=12, Wn=w, btype='bandpass', analog=False)
        return b, a


    def ipf(self):
        """Infinite (impulse response) Peak Filter at 19kHz"""
        w = float(F_PILOT) / float(F_SAMPLE) * 2
        q = w / 5.0 * float(F_SAMPLE)        # Q = f/bw, BW = 5 Hz
        b, a = sig.iirpeak(w, q)
        return b, a


class Demod:
    """STEP 2: RF Demodulation to Baseband"""

    def __init__(self, data):
        self.data = data
        self.fm_real = []
        self.crr = []
        self.crr3 = []
        self.rds_bb = []
        self.rds_sym = []
        self.rds_I = []
        self.rds_Q = []
        self.rds_bit = []
        self.demod_fm()

    def demod_fm(self):
        """STEP 2.1: DEMODULATE FM"""
        fm_cmplx = self.data[1:] * nmp.conj(self.data[:-1])                     # 1:end, 0:end-1
        self.data = nmp.angle(fm_cmplx)
        self.clock_recovery()

    def clock_recovery(self):
        """STEP 2.2: RECOVER RF CARRIER"""
        self.crr = sig.lfilter(filters.ipf()[0], filters.ipf()[1], self.data)   # filter out 19 kHz
        self.crr3 = sig.hilbert(self.crr) ** 3.0                                # multiply up to 57 kHz
        #self.crr3 = nmp.exp(nmp.arange(0,len(self.data))*1j*F_PILOT*3)          # free-running clock
        self.cull_rds()

    def cull_rds(self):
        """STEP 2.3: FILTER, MIX DOWN TO BB & DECIMATE"""
        rds_57 = sig.lfilter(filters.bpf()[0], filters.bpf()[1], self.data)     # BPF
        self.rds_bb = 1000 * rds_57 * self.crr3                                 # mix signal down by 57 kHz
        self.rds_bb = sig.decimate(self.rds_bb, DEC_RATE, zero_phase=True)      # Decimate
        self.pulse_shape()

    def pulse_shape(self):
        """STEP 2.4: APPLY R.R.COSINE FILTER"""
        self.rds_sym = 5 * sig.lfilter(filters.rrc(), 1, self.rds_bb)
        self.rds_I = nmp.real(self.rds_sym)
        self.rds_Q = nmp.imag(self.rds_sym)
        self.rds_phz = nmp.arctan2(self.rds_I, self.rds_Q)

    def bit_sync(self):
        return None

filters = Filter()

live = False

if live:
    sdr = RtlSdr()
    smp_rtl = rtl_sample()                              # RTL Samples

    with open('fm_data', 'wb') as f:
        pickle.dump(smp_rtl, f)
else:
    with open('fm_data', 'rb') as f:
        smp_rtl = pickle.load(f)


demod = Demod(smp_rtl)


phz = 4
symR = nmp.real(demod.rds_sym[phz::sps])
symI = nmp.imag(demod.rds_sym[phz::sps])

ang_sps = demod.rds_phz[phz::sps]
sym = (ang_sps > 0)

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

n=int(0)
for i in block_sync:
    if i[0] == 26:
        n += 1
        print i[1]


print(n, N_SAMPLES/DEC_RATE/sps/26)
# PLOTS
plt.figure()
plt.plot(symI, symR, 'b.', alpha=0.5)
#plt.scatter(demod.rds_I, demod.rds_Q)

plt.figure()
plt.plot(((demod.rds_sym[6000:9160])),'go')
plt.plot(nmp.real((demod.rds_sym[6000:9160])),'b.')
plt.plot((demod.rds_phz[6000:9160]),'r+')

#n1 = sps*1000
#n2 = n1 + sps*10
#f, (p1, p2, p3, p4, p5) = plt.subplots(5)
#p1.plot((demod.fm_real[n1*DEC_RATE:n2*DEC_RATE]))
#p2.plot(demod.rds_bb[n1:n2])
#p3.plot(demod.rds_sym[n1:n2])
#p4.plot(demod.rds_phz[n1:n2])
##p4.plot(nmp.arange(phz,sps*10+phz,sps),ang_sps[n1/sps:n2/sps],'r+')
#p5.plot(nmp.real(demod.crr[n1*DEC_RATE:n2*DEC_RATE]))
#
#plt.figure()
#plt.semilogx(w/2/nmp.pi,20*nmp.log10(abs(h)))
#plt.psd(smp_bpc[phz::sps], NFFT=2048, Fs=F_SAMPLE/1000/DEC_RATE, Fc=0)
#plt.psd(smp_bpc, NFFT=2048, Fs=F_SAMPLE/1000/DEC_RATE, Fc=0)
#plt.psd(pilot*100, NFFT=2048, Fs=F_SAMPLE/1000, Fc=0)
plt.show()
