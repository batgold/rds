#!/usr/bin/env python

from rtlsdr import RtlSdr
import commpy as com
import matplotlib.pyplot as plt
import numpy as nmp
import scipy.signal as sig
sdr = RtlSdr()

f_sample = int(228e3)       # 225-300 kHz and 900 - 2032 kHz
f_center = int(96.5e6)      # FM Station, need strong RDBS signal, SF
#f_center = int(97.7e6)      # FM Station, need strong RDBS signal, PAL
f_pilot = int(19e3)         # pilot tone, 19kHz from Fc
n_samples = int(512*512*2)  # must be multiple of 512, should be a multiple of 16384 (URB size)
dec_rate = 12               # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
f_sym = 1187.5              # Symbol rate, full bi-phase
sps = int(f_sample/dec_rate/f_sym)   # Samples per Symbol

def rtl_sample():
    sdr.sample_rate = f_sample
    sdr.center_freq = f_center
    sdr.gain = 'auto'
    smp_rtl = sdr.read_samples(n_samples)
    sdr.close()
    return smp_rtl[2000:]               #first 2000 samples are bogus

def demod_fm(smp):
    smp_fm_cmplx = smp[1:] * nmp.conj(smp[:-1])  # 1:end, 0:end-1
    smp_fm = nmp.angle(smp_fm_cmplx)
    return smp_fm

def build_ipf():            #Infinite (impulse response) Peak Filter
    pilot_wo = float(f_pilot)/float(f_sample)*2
    pilot_Q = pilot_wo / 5.0 * float(f_sample)  # Q = f/bw, BW = 5 Hz
    b,a = sig.iirpeak(pilot_wo, pilot_Q)
    return b,a

def build_bpf():            #BPF @ 57 kHz
    Wn = [(f_pilot*3 - 3e3)/f_sample*2, (f_pilot*3 + 3e3)/f_sample*2]   #slightly wider (3kHz)
    b,a = sig.butter(N=12, btype='bandpass', Wn=Wn, analog=False)
    return b,a

def build_rrc():            #Root-Raised Cosine Filter
    N = int(30)
    T = 1/1187.5/2
    alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
    #t,rrc = com.filters.rrcosfilter(N,alfa,T,f_sample/dec_rate)
    rrc = [-0.00281562025759368,-0.00243888516957360,-0.000999844542191435,0.00107593094803098,0.00303973034012925,0.00406700703874643,0.00357325036472412,0.00148757358716287,-0.00162769040855969,-0.00468282782128020,-0.00639101106088724,-0.00573885664637510,-0.00244729848210667,0.00275023551791118,0.00815158917037663,0.0115038199095970,0.0107291667736578,0.00477805894125585,-0.00564522027360720,-0.0177416940767021,-0.0268422464557264,-0.0272355771946698,-0.0134654388344483,0.0181901542149565,0.0684322485815653,0.134211232278632,0.208806091825802,0.282774215523415,0.345612930084173,0.387782741962203,0.402633696835896,0.387782741962203,0.345612930084173,0.282774215523415,0.208806091825802,0.134211232278632,0.0684322485815653,0.0181901542149565,-0.0134654388344483,-0.0272355771946698,-0.0268422464557264,-0.0177416940767021,-0.00564522027360720,0.00477805894125585,0.0107291667736578,0.0115038199095970,0.00815158917037663,0.00275023551791118,-0.00244729848210667,-0.00573885664637510,-0.00639101106088724,-0.00468282782128020,-0.00162769040855969,0.00148757358716287,0.00357325036472412,0.00406700703874643,0.00303973034012925,0.00107593094803098,-0.000999844542191435,-0.00243888516957360,-0.00281562025759368]
    return rrc

def clock_recovery(smp):
    crr = sig.lfilter(ipf[0], ipf[1], smp)  # filter out 19 kHz
    crr3 = sig.hilbert(crr) ** 3.0             # multiply up to 57 kHz
    return crr, crr3

def mix_and_decimate(smp):
    smp_mix = smp * crr3         # mix signal down by 57 kHz
    smp_bb = sig.decimate(smp_mix, dec_rate, zero_phase=True)  # decimate
    return smp_bb*1000

ipf = build_ipf()
bpf = build_bpf()
rrc = build_rrc()

smp_rtl = rtl_sample()                              # RTL Samples
smp_fm = demod_fm(smp_rtl)                          # FM Demodulate
smp_rbds = sig.lfilter(bpf[0], bpf[1], smp_fm)      # Bandpass filter RBDS

crr,crr3 = clock_recovery(smp_fm)                   # Pilot tone extraction

smp_bb = mix_and_decimate(smp_rbds)        # Mix down to baseband
smp_bb2 = sig.lfilter(rrc,1,smp_bb)

#sym_bit = (smp_bb2 > 0)
print("Samples/sec", str(sps))
ang = nmp.arctan2(nmp.real(smp_bb2),nmp.imag(smp_bb2))

phz =7
symR = nmp.real(smp_bb2[phz::sps])
symI = nmp.imag(smp_bb2[phz::sps])

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

print(n, n_samples/dec_rate/sps/26)
# PLOTS
plt.figure()
plt.scatter(symR,symI)

n1 = sps*1000
n2 = n1 + sps*10
f, (p1, p2, p3, p4, p5) = plt.subplots(5)
p1.plot((smp_rbds[n1*dec_rate:n2*dec_rate]))
p2.plot(smp_bb[n1:n2])
p3.plot(smp_bb2[n1:n2])
p4.plot(ang[n1:n2])
p4.plot(nmp.arange(phz,sps*10+phz,sps),ang_sps[n1/sps:n2/sps],'r+')
p5.plot(nmp.real(crr[n1:n2]))

#plt.figure()
#plt.semilogx(w/2/nmp.pi,20*nmp.log10(abs(h)))
#plt.psd(smp_bb2[phz::sps], NFFT=2048, Fs=f_sample/1000/dec_rate, Fc=0)
#plt.psd(smp_bb2, NFFT=2048, Fs=f_sample/1000/dec_rate, Fc=0)
#plt.psd(pilot*100, NFFT=2048, Fs=f_sample/1000, Fc=0)
plt.show()
