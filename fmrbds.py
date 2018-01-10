#!/usr/bin/env python

import pickle
import commpy as com
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig
from sys import stdout
from time import sleep
from bitarray import bitarray
from rtlsdr import RtlSdr

F_SAMPLE = int(228e3)       # 225-300 kHz and 900 - 2032 kHz
#F_CENTER = int(96.5e6)      # FM Station, need strong RDBS signal, SF (KOIT XMAS)
#F_CENTER = int(97.7e6)      # FM Station, PAL
F_CENTER = int(104.5e6)      # FM Station, WV
F_PILOT = int(19e3)         # pilot tone, 19kHz from Fc
N_SAMPLES = int(512*512*3)  # must be multiple of 512, should be a multiple of 16384 (URB size)
DEC_RATE = int(12)          # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
F_SYM = 1187.5            # Symbol rate, full bi-phase
SPS = int(F_SAMPLE/DEC_RATE/F_SYM)   # Samples per Symbol


def rtl_sample():
    """STEP 0: SAMPLE DATA FROM RTLSDR"""
    print("READING DATA")
    sdr.sample_rate = F_SAMPLE
    sdr.center_freq = F_CENTER
    sdr.gain = 'auto'
    smp_rtl = sdr.read_samples(N_SAMPLES)
    sdr.close()
    return smp_rtl[2000:]               #first 2000 samples are bogus


class Filters:
    """STEP 1: CONSTRUCT FILTERS"""

    def __init__(self):
        print("BUILDING FILTERS")
        self.rrc = self.build_rrc()
        self.bpf = self.build_bpf()
        self.ipf = self.build_ipf()
        self.clk = self.build_clk()

    def build_rrc(self):
        """Cosine Filter"""
        N = int(120)
        T = 1/1187.5*3/8
        alfa = 1 #put 8 samples per sym period. i.e. 16+1 in the main lobe
        __,rrc = com.filters.rrcosfilter(N,alfa,T,F_SAMPLE/DEC_RATE)
        return rrc

    def build_bpf(self):
        """Bandpass Filter, at 57kHz"""
        cutoff = 3.0e3          # one-sided cutoff freq, slightly larger than 2.4kHz
        w = [(F_PILOT*3 - cutoff) / F_SAMPLE*2, (F_PILOT*3 + cutoff) / F_SAMPLE*2]
        b, a = sig.butter(N=12, Wn=w, btype='bandpass', analog=False)
        return b, a

    def build_ipf(self):
        """Infinite (impulse response) Peak Filter at 19kHz"""
        w = F_PILOT / float(F_SAMPLE / 2.0)
        q = w / 5.0 * F_SAMPLE        # Q = f/bw, BW = 5 Hz
        b, a = sig.iirpeak(w, q)
        return b, a

    def build_clk(self):
        """Infinite (impulse response) Peak Filter at Symbol Rate 1187.5Hz"""
        #w = F_SYM / (float(F_SAMPLE) / float(DEC_RATE) / 2.0)
        w = (F_PILOT*3 + F_SYM) / float(F_SAMPLE / 2.0)
        #q = w / 5.0 * F_SAMPLE * DEC_RATE        # Q = f/bw, BW = 5 Hz
        q = w / 5.0 * F_SAMPLE          # Q = f/bw, BW = 5 Hz
        b, a = sig.iirpeak(w, q)
        return b, a


class Demod:
    """STEP 2: RF Demodulation to Baseband"""

    def __init__(self, data, phz_offset):
        print("DEMODULATING")
        self.data = data
        self.phz_offset = phz_offset
        self.fm = []
        self.crr = []
        self.clk = []
        self.clk_len = []
        self.bb = []
        self.I = []
        self.Q = []
        self.sym = []
        self.sym2 = []
        self.bits = []
        self.demod_fm()
        self.carrier_recovery()
        self.cull_rds()
        self.pulse_shape()
        self.clock_recovery()
        self.symbol_decode()
        self.bit_decode()

    def demod_fm(self):
        """STEP 2.1: DEMODULATE FM"""
        print("...DEMOD FM")
        x1 = self.data[1:] * nmp.conj(self.data[:-1])                     # 1:end, 0:end-1
        self.fm = nmp.angle(x1)

    def carrier_recovery(self):
        """STEP 2.2: RECOVER RF CARRIER"""
        print("...RECOVER CARRIER")
        y1 = sig.lfilter(filter.ipf[0], filter.ipf[1], self.fm)   # filter out 19 kHz
        #self.crr3 = nmp.exp(nmp.arange(0,len(self.data))*1j*F_PILOT*3)          # free-running clock
        self.crr = sig.hilbert(y1) ** 3.0                                # multiply up to 57 kHz

    def cull_rds(self):
        """STEP 2.3: FILTER, MIX DOWN TO BB & DECIMATE"""
        print("...MOVE TO BASEBAND")
        x2 = sig.lfilter(filter.bpf[0], filter.bpf[1], self.fm)     # BPF
        x3 = 1000 * x2 * self.crr                                 # mix signal down by 57 kHz
        #self.bb = x3[::DEC_RATE]
        self.bb = sig.decimate(x3, DEC_RATE, zero_phase=True)      # Decimate
        self.bb = self.bb[SPS*2:]

    def pulse_shape(self):
        """STEP 2.4: APPLY R.R.COSINE FILTER"""
        print("...PULSE SHAPE")
        x4 = 90 * sig.lfilter(filter.rrc, 1, self.bb)     #gain of 5 seems gud
        self.I = nmp.real(x4)
        self.Q = nmp.imag(x4)
        self.amp = self.I
        self.phz = nmp.arctan2(self.I, self.Q)

    def clock_recovery(self):
        """STEP 2.5: DETERMINE SYMBOL CLOCK RATE"""
        print("...RECOVER CLOCK")
        #x6 = 1000 * sig.lfilter(filter.clk[0], filter.clk[1], self.bb)
        x6 = sig.lfilter(filter.clk[0], filter.clk[1], self.fm)
        x7 = 1000 * x6 * self.crr
        x8 = nmp.real(x7)
        x8 = sig.decimate(x7, DEC_RATE, zero_phase=True)
        x8 = x8[SPS*2:]
        x9 = (x8 > 0)
        #x10 = x9[1:]
        self.clk = x9

    def symbol_decode(self):
        """STEP 2.6: SAMPLE SYMBOLS AT CLOCK RATE"""
        print("...DECODING SYMBOLS")
        #n = 459*16
        n = 0
        while n < 0:
        #while n < len(self.amp) - SPS:
        #while n < 459*16+16*7:
            if self.clk[n] == 1:       #capture signal at clock ON
                symbol = []
                m = 0
                while self.clk[n+m] == 1:
                    symbol.append(self.amp[n+m])
                    m += 1
                symbol.remove(max(symbol))
                symbol.remove(min(symbol))
                symbol = nmp.sum(symbol)
                #self.sym.append(symbol/80)
                self.sym.append((symbol > 0))
                self.clk_len.append(m)
                n += m
            else:
                n += 1                  #run through OFF

        x5 = self.phz[self.phz_offset::SPS]
        self.I = self.I[self.phz_offset::SPS]
        self.Q = self.Q[self.phz_offset::SPS]
        self.sym = (x5 > 0)

    def bit_decode(self):
        """STEP 2.7: DECODE MANCHESTER SYMBOLS"""
        self.bits = nmp.bitwise_xor(self.sym[1:], self.sym[:-1])


class Block_Sync:
    """STEP 3: Synchronize and Interrogate Blocks"""

    def __init__(self, bits):
        print("SYNCHRONIZING BLOCKS")
        self.bits = bits
        self.parity = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
        self.syndromes = [984, 980, 604, 600]
        self.word_list = ['A', 'B', 'C', 'D']
        self.words, self.words_indx, self.words_dist = self.get_words()
        self.blocks, self.blocks_indx = self.get_blocks()
        self.group_indx = self.get_groups()
        print "...WORDS: ", len(self.words)
        print "...BLOCKS:", len(self.blocks)
        print "...GROUPS:", len(self.group_indx)

    def get_syndrome(self, n_bit):
        syndrome = 0;
        for n in range(26):
            if self.bits[n_bit + n]:
                syndrome = nmp.bitwise_xor(syndrome, self.parity[n])    #multiply word by H
        return syndrome

    def get_words(self):
        prev_bit = 0
        words = []
        words_indx = []
        words_dist = []

        for n_bit in range(len(self.bits) - 26):
            syndrome = self.get_syndrome(n_bit)         #calculate syndrome

            if syndrome in self.syndromes:              #if valid syndrome
                indx = self.syndromes.index(syndrome)
                words.append(self.word_list[indx])
                words_indx.append(n_bit)
                words_dist.append(n_bit - prev_bit)
                prev_bit = n_bit
        return words, words_indx, words_dist

    def get_blocks(self):
        blocks = []
        blocks_indx = []
        for n, word in enumerate(self.words):
            if word == 'A' or self.words_dist[n] == 26:
                blocks.append(word)
                blocks_indx.append(self.words_indx[n])
        return blocks, blocks_indx

    def get_groups(self):
        group_indx = []
        for n in range(len(self.blocks) - 4):
            group = self.blocks[n:n+4]
            if nmp.array_equal(group, self.word_list):
                group_indx.append(self.blocks_indx[n])
        return group_indx


class Decode:
    """STEP 4: Decode Bits into Message"""

    def __init__(self, bits, bit_start):
        self.bits = bits
        self.ps = ['_','_','_','_''_','_','_','_']
        self.rt = ['_','_','_','_','_','_','_','_','_','_','_','_','_','_','_','_']

        for bit in bit_start:
            self.set_blocks(bit)
            pi = self.prog_id()
            gt = self.group_type()
            pt = self.prog_type()

            if gt == '0A':
                self.cc = self.char_code()
                self.prog_service()
            elif gt == '2A':
                self.cc = self.text_segment()
                self.radiotext()

            print pi, gt, pt, self.cc, ''.join(self.ps), ' / ', ''.join(self.rt)
            print self.C
            print self.D

    def set_blocks(self, bit):
        self.A = 1 * self.bits[bit     :bit + 16]
        self.B = 1 * self.bits[bit + 26:bit + 42]
        self.C = 1 * self.bits[bit + 52:bit + 68]
        self.D = 1 * self.bits[bit + 78:bit + 94]

    def prog_id(self):
        """Program Identification"""
        K = 4096
        W = 21672
        pi_int = self.bit2int(self.A)
        if pi_int >= W:
            pi0 = 'W'
            pi1 = W
        else:
            pi0 = 'K'
            pi1 = K
        pi2 = nmp.floor_divide(pi_int - pi1, 676)
        pi3 = nmp.floor_divide(pi_int - pi1 - pi2*676, 26)
        pi4 = pi_int - pi1 - pi2*676 - pi3*26
        pi = pi0 + chr(pi2+65) + chr(pi3+65) + chr(pi4+65)
        return pi

    def group_type(self):
        """Group Type"""
        gt_num = self.bit2int(self.B[0:4])
        gt_ver = self.B[4]
        gt = str(gt_num) + chr(gt_ver + 65)      # 5th bit = Version (A|B)
        return gt

    def prog_type(self):
        """Program Type"""
        pt_codes = ['None', 'News', 'Info', 'Sports', 'Talk', 'Rock', \
                    'Classic Rock', 'Adult Hits', 'Soft Rock', \
                    'Top 40', 'Country', 'Oldies', 'Soft', 'Nostalgia', 'Jazz', \
                    'Classical', 'R&B', 'Foreign Lang.', 'Religious Music', \
                    'Religious Talk', 'Personality', 'Public', 'College', 'NA', \
                    'Weather', 'Emergency Test', 'ALERT!']
        pt_num = self.bit2int(self.B[6:11])
        pt = pt_codes[pt_num]
        return pt

    def char_code(self):
        """Character Placement"""
        return self.bit2int(self.B[-2:])

    def text_segment(self):
        """Character Placement"""
        return self.bit2int(self.B[-4:])

    def prog_service(self):
        ps1 = self.bit2int(self.D[0:8])
        ps2 = self.bit2int(self.D[8:16])
        self.ps[self.cc] = chr(ps1) + chr(ps2)

    def radiotext(self):
        rt1 = self.bit2int(self.C[0:8])
        rt2 = self.bit2int(self.C[8:16])
        rt3 = self.bit2int(self.D[0:8])
        rt4 = self.bit2int(self.D[8:16])

        self.rt[self.cc] = chr(rt1) + chr(rt2) + chr(rt3) + chr(rt4)

    def bit2int(self, bits):
        """Convert bit string to integer"""
        word = 0
        for bit in bits:
            word = (word<<1) | bit
        return int(word)

    def print_msg(self):
        return None


class Plot:
    """PLOT"""

    def __init__(self, data):
        self.amp = data.amp
        self.phz = data.phz
        self.I = data.I
        self.Q = data.Q
        self.sym = data.sym
        self.bit = data.bits
        self.clk = data.clk

    def constellation(self):
        plt.figure()
        plt.plot(self.I, self.Q, 'b.', alpha=0.5)
        plt.show()

    def time_domain(self, T1):
        T2 = T1 + SPS*10
        T = nmp.arange(T1,T2)
        plt.figure()
        plt.plot(T, self.amp[T1:T2], 'b.')
        plt.plot(T, self.phz[T1:T2], 'r+')
        plt.plot(T, self.clk[T1:T2], 'g')
        plt.show()

    def clock(self):
        plt.figure()
        plt.plot(self.clk)
        plt.show()

    def fm(self, fm):
        plt.figure()
        plt.psd(fm, NFFT=2048, Fs=F_SAMPLE/1000)
        plt.show()

    def bb(self, bb):
        plt.figure()
        plt.psd(bb, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        plt.ylim([-25, 0])
        plt.yticks(nmp.arange(-25, 0, 5))
        plt.show()


live = True

if live:
    sdr = RtlSdr()
    smp_rtl = rtl_sample()                              # RTL Samples

    with open('fm_data', 'wb') as f:
        pickle.dump(smp_rtl, f)
else:
    print("READING DATA")
    with open('fm_data', 'rb') as f:
        smp_rtl = pickle.load(f)


filter = Filters()
data = Demod(smp_rtl, 4)

group_indx = Block_Sync(data.bits).group_indx

Decode(data.bits, group_indx)

quit()
plots = Plot(data)
plots.time_domain(500)
quit()
plots.fm(data.fm)
plots.fm(data.bb)
plots.constellation()


plots.bb(data.bb)

if groups:
    plots.time_domain(groups[0])
    #plots.clock()

#if plot:
    #ca = nmp.asarray(data.clk_len[100:])
    #gt = (ca > 8).sum()
    #lt = (ca < 8).sum()

    #print("GT", gt)
    #print("LT", lt)
