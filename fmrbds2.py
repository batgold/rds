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

LIVE = True

F_SAMPLE = int(228e3)       # 225-300 kHz and 900 - 2032 kHz
F_CENTER = int(101.3e6)     # FM Station
F_PILOT = int(19e3)         # pilot tone, 19kHz from Fc
N_SAMPLES = int(512*512*3)  # must be multiple of 512, should be a multiple of 16384 (URB size)
DEC_RATE = int(12)          # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
F_SYM = 1187.5            # Symbol rate, full bi-phase
SPS = int(F_SAMPLE/DEC_RATE/F_SYM)   # Samples per Symbol

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
        w = F_SYM / float(F_SAMPLE / DEC_RATE / 2.0)
        q = w / 5.0 * F_SAMPLE * DEC_RATE         # Q = f/bw, BW = 5 Hz
        b, a = sig.iirpeak(w, q)
        return b, a


class Radio_Data_System():

    def __init__(self, sdr, filters, code, bit_o):
        self.sdr = sdr
        self.filters = filters
        self.code = code
        self.bit_o = bit_o

    def start(self):
        if LIVE:
            self.read_samples()
        else:
            self.load_samples()

    def read_samples(self):
        print("READING DATA")
        samples = self.sdr.read_samples(N_SAMPLES)
        samples = samples[2000:]        #first 2000 samples are bogus
        self.demodulate(samples)

        with open('radio_signal', 'wb') as f:
            pickle.dump(samples, f)

    def load_samples(self):
        print "LOADING DATA"
        with open('radio_signal', 'rb') as f:
            samples = pickle.load(f)
        self.demodulate(samples)

    def demodulate(self, samples):
        print "DEMODULATE FM"
        x1 = samples[1:] * nmp.conj(samples[:-1])    # 1:end, 0:end-1
        x2 = nmp.angle(x1)
        x3 = sig.lfilter(self.filters.bpf[0], self.filters.bpf[1], x2) # BPF
        crr = self.recover_carrier(x2)  # get 57kHz carrier
        x4 = 1000 * crr * x3        # mix down to baseband
        x5 = sig.decimate(x4, DEC_RATE, zero_phase=True)
        x6 = 100 * sig.lfilter(self.filters.rrc, 1, x5)
        amp = nmp.absolute(x6)
        phz = nmp.angle(x6)
        clk = self.recover_clock(x5)
        sym = self.recover_symbols(amp, phz, clk)
        bits = nmp.bitwise_xor(sym[1:], sym[:-1])
        self.synchronize(bits)

    def recover_carrier(self, samples):
        print "RECOVER CARRIER"
        crr = sig.lfilter(self.filters.ipf[0], self.filters.ipf[1], samples)
        return sig.hilbert(crr) ** 3.0

    def recover_clock(self, samples):
        print "RECOVER CLOCK"
        clk = sig.lfilter(self.filters.clk[0], self.filters.clk[1], samples)
        return (clk > 0)

    def recover_symbols(self, amp, phz, clk):
        print "RECOVER SYMBOLS"
        toc = 0
        m = 0
        bit_amp = []
        bit_phz = []
        for n, tic in enumerate(clk):
            if abs(1*tic - toc):
                toc = tic
                m = n
                bit_amp.append(amp[n])
                bit_phz.append(phz[n])
        x = nmp.asarray(bit_phz)  #use phase
        x = (x > 0)
        self.I = bit_amp[self.bit_o::2] * nmp.cos(bit_phz[self.bit_o::2])
        self.Q = bit_amp[self.bit_o::2] * nmp.sin(bit_phz[self.bit_o::2])
        return x[self.bit_o::2]

    def synchronize(self, bits):
        print "SYNCHRONIZE BITS"
        words = []
        blocks = []
        groups = []
        m = 0

        for n, bit in enumerate(bits[:-26]):
            syndrome = self.code.syndrome(bits, n)

            if syndrome in self.code.syndromes:
                i = self.code.syndromes.index(syndrome)     # 1,2,3,4 = A,B,C,D
                words.append([i, n, n - m])   # [letter, location, distance from prev word]
                m = n

        for n, word in enumerate(words):
            if (word[0] - words[n-1][0] == 1 and word[2] == 26) or word[0] == 0:
                blocks.append(word)

        for n, block in enumerate(blocks[:-4]):
            group = [block[0] for block in blocks[n:n+4]]
            if nmp.array_equal(group, self.code.words):
                groups.append(block[1])

        print "...WORDS: ", len(words)
        print "...BLOCKS:", len(blocks)
        print "...GROUPS:", len(groups)

        decode(bits, groups)

    def decode(self, bits, groups)
        Decode(bit, groups)


class Code:
    """CODE PARAMETERS"""

    def __init__(self):
        self.parity = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
        self.syndromes = [984, 980, 604, 600]
        #self.words = ['A', 'B', 'C', 'D']
        self.words = [0,1,2,3]

    def syndrome(self, bits, bit_n):
        syn = 0
        for n in range(26):
            if bits[bit_n + n]:
                syn = nmp.bitwise_xor(syn, self.parity[n])
        return syn

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
            else:
                self.cc = 0

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
        self.data = data
        self.amp = data.amp
        self.phz = data.phz
        self.sym = data.sym
        self.bit = data.bits
        self.clk = data.clk

    def constellation(self):
        plt.figure()
        plt.plot(self.data.I, self.data.Q, 'b.', alpha=0.5)
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

    def fm(self):
        plt.figure()
        plt.psd(self.data.fm, NFFT=2048, Fs=F_SAMPLE/1000)
        plt.psd(self.clk, NFFT=2048, Fs=F_SAMPLE/1000)
        plt.ylim([-50, 0])
        plt.yticks(nmp.arange(-50, 0, 10))
        plt.show()

    def bb(self):
        plt.figure()
        plt.psd(self.data.bb, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        plt.psd(self.clk, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        plt.ylim([-25, 0])
        plt.yticks(nmp.arange(-25, 0, 5))
        plt.show()


def main():
    filters = Filters()
    code = Code()
    sdr = RtlSdr()
    rds = Radio_Data_System(sdr, filters, code, 3)

    sdr.rs = F_SAMPLE
    sdr.fc = F_CENTER
    sdr.gain = 'auto'

    rds.start()

    sdr.close()

main()

quit()


plots = Plot(data)
plots.constellation()
#plots.time_domain(800)
#plots.bb()

