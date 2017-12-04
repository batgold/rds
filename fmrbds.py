#!/usr/bin/env python

import commpy as com
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig
from rtlsdr import RtlSdr

sdr = RtlSdr()

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
    sdr.sample_rate = F_SAMPLE
    sdr.center_freq = F_CENTER
    sdr.gain = 'auto'
    smp_rtl = sdr.read_samples(N_SAMPLES)
    sdr.close()
    return smp_rtl[2000:]               #first 2000 samples are bogus


class Demod:
    """RF Demodulation to Baseband"""

    def __init__(self,data):
        self.data = []

    def demod_fm(smp):
        fm_cmplx = smp[1:] * nmp.conj(smp[:-1])  # 1:end, 0:end-1
        return nmp.angle(fm_cmplx)


    def clock_recovery(smp):
        crr = sig.lfilter(filters.ipf()[0], filters.ipf()[1], smp)      # filter out 19 kHz
        crr3 = sig.hilbert(crr) ** 3.0              # multiply up to 57 kHz
        return crr, crr3


    def mix_and_decimate(smp):
        smp_mix = smp * crr3         # mix signal down by 57 kHz
        smp_bb = sig.decimate(smp_mix, DEC_RATE, zero_phase=True)  # decimate
        return smp_bb*1000


class Filter:
    """Construct Filters"""

    def rrc(self):
        """Cosine Filter"""
        rrc = [-0.00281562025759368,-0.00243888516957360,-0.000999844542191435,0.00107593094803098,0.00303973034012925,0.00406700703874643,0.00357325036472412,0.00148757358716287,-0.00162769040855969,-0.00468282782128020,-0.00639101106088724,-0.00573885664637510,-0.00244729848210667,0.00275023551791118,0.00815158917037663,0.0115038199095970,0.0107291667736578,0.00477805894125585,-0.00564522027360720,-0.0177416940767021,-0.0268422464557264,-0.0272355771946698,-0.0134654388344483,0.0181901542149565,0.0684322485815653,0.134211232278632,0.208806091825802,0.282774215523415,0.345612930084173,0.387782741962203,0.402633696835896,0.387782741962203,0.345612930084173,0.282774215523415,0.208806091825802,0.134211232278632,0.0684322485815653,0.0181901542149565,-0.0134654388344483,-0.0272355771946698,-0.0268422464557264,-0.0177416940767021,-0.00564522027360720,0.00477805894125585,0.0107291667736578,0.0115038199095970,0.00815158917037663,0.00275023551791118,-0.00244729848210667,-0.00573885664637510,-0.00639101106088724,-0.00468282782128020,-0.00162769040855969,0.00148757358716287,0.00357325036472412,0.00406700703874643,0.00303973034012925,0.00107593094803098,-0.000999844542191435,-0.00243888516957360,-0.00281562025759368]
        #N = int(30)
        #T = 1/1187.5/2
        #alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
        #t,rrc = com.filters.rrcosfilter(N,alfa,T,F_SAMPLE/DEC_RATE)
        return rrc


    def bpf(self):
        """Bandpass Filter, at 57kHz"""
        #slightly wider (3kHz)
        w = [(F_PILOT*3 - 3e3) / F_SAMPLE*2, (F_PILOT*3 + 3e3) / F_SAMPLE*2]
        b, a = sig.butter(N=12, btype='bandpass', Wn=w, analog=False)
        return b,a


    def ipf(self):
        """Infinite (impulse response) Peak Filter at 19kHz"""
        w = float(F_PILOT) / float(F_SAMPLE) * 2
        q = w / 5.0 * float(F_SAMPLE)        # Q = f/bw, BW = 5 Hz
        b, a = sig.iirpeak(w, q)
        return b, a


filters = Filter()

smp_rtl = rtl_sample()                              # RTL Samples

bb = Demod(smp_rtl)
quit()

smp_fm = demod_fm(smp_rtl)                          # FM Demodulate
smp_rbds = sig.lfilter(filters.bpf()[0], filters.bpf()[1], smp_fm)      # Bandpass filter RBDS

crr,crr3 = clock_recovery(smp_fm)                   # Pilot tone extraction

smp_bb = mix_and_decimate(smp_rbds)                 # Mix down to baseband
smp_bpc = sig.lfilter(filters.rrc(),1,smp_bb)                 # Bi-Phase Code

#sym_bit = (smp_bpc > 0)
ang = nmp.arctan2(nmp.real(smp_bpc),nmp.imag(smp_bpc))

phz = 7
symR = nmp.real(smp_bpc[phz::sps])
symI = nmp.imag(smp_bpc[phz::sps])

ang_sps = ang[phz::sps]
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
plt.scatter(symR,symI)

n1 = sps*1000
n2 = n1 + sps*10
f, (p1, p2, p3, p4, p5) = plt.subplots(5)
p1.plot((smp_rbds[n1*DEC_RATE:n2*DEC_RATE]))
p2.plot(smp_bb[n1:n2])
p3.plot(smp_bpc[n1:n2])
p4.plot(ang[n1:n2])
p4.plot(nmp.arange(phz,sps*10+phz,sps),ang_sps[n1/sps:n2/sps],'r+')
p5.plot(nmp.real(crr[n1*DEC_RATE:n2*DEC_RATE]))

#plt.figure()
#plt.semilogx(w/2/nmp.pi,20*nmp.log10(abs(h)))
#plt.psd(smp_bpc[phz::sps], NFFT=2048, Fs=F_SAMPLE/1000/DEC_RATE, Fc=0)
#plt.psd(smp_bpc, NFFT=2048, Fs=F_SAMPLE/1000/DEC_RATE, Fc=0)
#plt.psd(pilot*100, NFFT=2048, Fs=F_SAMPLE/1000, Fc=0)
plt.show()
