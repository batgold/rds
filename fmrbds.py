#!/usr/bin/env python

import sys
import pickle
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.integrate import cumtrapz
from filters import Filters
from rtlsdr import RtlSdr


class Code:
    """CODE PARAMETERS"""

    def __init__(self):
        self.parity = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
        self.syndromes = [984, 980, 604, 600]
        #self.words = ['A', 'B', 'C', 'D']
        self.words = [0,1,2,3]
        self.pi = ''
        self.gt = ''
        self.pt = ''
        self.ps = ['_','_','_','_','_','_','_','_']
        self.rt = ['_','_','_','_','_','_','_','_','_','_','_','_','_','_','_','_']

    def print_code(self):
        print self.pi, self.gt, self.pt, ''.join(self.ps), ' / ', ''.join(self.rt)

    def syndrome(self, bits, bit_n):
        syn = 0
        for n in range(26):
            if bits[bit_n + n]:
                syn = nmp.bitwise_xor(syn, self.parity[n])
        return syn

    def prog_id(self, A):
        """Program Identification"""
        K = 4096
        W = 21672
        pi_int = self.bit2int(A)
        if pi_int >= W:
            pi0 = 'W'
            pi1 = W
        else:
            pi0 = 'K'
            pi1 = K
        pi2 = nmp.floor_divide(pi_int - pi1, 676)
        pi3 = nmp.floor_divide(pi_int - pi1 - pi2*676, 26)
        pi4 = pi_int - pi1 - pi2*676 - pi3*26
        self.pi = pi0 + chr(pi2+65) + chr(pi3+65) + chr(pi4+65)

    def group_type(self, B):
        """Group Type"""
        gt_num = self.bit2int(B[0:4])
        gt_ver = B[4]
        self.gt = str(gt_num) + chr(gt_ver + 65)      # 5th bit = Version (A|B)

    def prog_type(self, B):
        """Program Type"""
        pt_codes = ['None', 'News', 'Info', 'Sports', 'Talk', 'Rock', \
                    'Classic Rock', 'Adult Hits', 'Soft Rock', \
                    'Top 40', 'Country', 'Oldies', 'Soft', 'Nostalgia', 'Jazz', \
                    'Classical', 'R&B', 'Foreign Lang.', 'Religious Music', \
                    'Religious Talk', 'Personality', 'Public', 'College', 'NA', \
                    'Weather', 'Emergency Test', 'ALERT!']
        pt_num = self.bit2int(B[6:11])
        self.pt = pt_codes[pt_num]

    def prog_service(self, B, D):
        cc = self.bit2int(B[-2:])
        ps1 = self.bit2int(D[0:8])
        ps2 = self.bit2int(D[8:16])
        self.ps[2*cc] = chr(ps1) + chr(ps2)

    def radiotext(self, B, C, D):
        cc = self.bit2int(B[-4:])
        rt1 = self.bit2int(C[0:8])
        rt2 = self.bit2int(C[8:16])
        rt3 = self.bit2int(D[0:8])
        rt4 = self.bit2int(D[8:16])
        self.rt[4*cc:4*cc+4] = chr(rt1) + chr(rt2) + chr(rt3) + chr(rt4)

    def bit2int(self, bits):
        """Convert bit string to integer"""
        word = 0
        for bit in bits:
            word = (word<<1) | bit
        return int(word)

class Plot:
    """PLOT"""

    def __init__(self):
        return None

    def constellation(self, I, Q):
        plt.figure()
        plt.plot(I, Q, 'b.', alpha=0.5)
        plt.show()

    def time_domain(self, amp, phz, clk):
        plt.figure()
        plt.plot(amp, 'b.')
        plt.plot(phz, 'r+')
        plt.plot(clk, 'g')
        plt.show()

    def fm(self, samples):
        plt.figure()
        plt.psd(samples, NFFT=2048, Fs=F_SAMPLE/1000)
        #plt.psd(self.clk, NFFT=2048, Fs=F_SAMPLE/1000)
        plt.ylim([-50, 0])
        plt.yticks(nmp.arange(-50, 0, 10))
        plt.show()

    def bb(self, samples):
        plt.figure()
        plt.psd(samples, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        #plt.psd(self.clk, NFFT=2048, Fs=F_SAMPLE/DEC_RATE/1000)
        plt.ylim([-25, 0])
        plt.yticks(nmp.arange(-25, 0, 5))
        plt.show()

class Radio_Data_System():
    """RADIO"""

    def __init__(self, filters, code, plot):
        self.filters = filters
        self.code = code
        self.plot = plot

    def start(self):
        if LIVE:
            self.read_samples()
        else:
            self.load_samples()

    def read_samples(self):
        print("READING DATA")
        self.sdr = RtlSdr()
        self.sdr.rs = F_SAMPLE
        self.sdr.fc = F_CENTER
        self.sdr.gain = 'auto'

        samples = self.sdr.read_samples(N_SAMPLES)
        samples = samples[2000:]        #first 2000 samples are bogus
        self.demodulate(samples)
        self.sdr.close()

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
        #self.plot.fm(x2)
        self.calc_snr(x2)
        crr = self.recover_carrier(x2)  # get 57kHz carrier
        x3 = sig.lfilter(self.filters.bpf[0], self.filters.bpf[1], x2) # BPF
        x4 = 1000 * crr * x3        # mix down to baseband
        x5 = sig.decimate(x4, DEC_RATE, zero_phase=True)
        #self.plot.bb(x5)
        clk = self.recover_clock(x5)
        x6 = 90 * sig.lfilter(self.filters.rrc, 1, x5)
        sym = self.recover_symbols(clk, x6)
        bits = nmp.bitwise_xor(sym[1:], sym[:-1])
        self.synchronize(bits)

    def calc_snr(self, x):
        [P, f] = plt.psd(x, NFFT=2048, Fs=F_SAMPLE)
        s_idx = (nmp.abs(f - (57e3 + F_SYM))).argmin()
        n_idx = (nmp.abs(f - 57e3)).argmin()
        S = 10*nmp.log10(P[s_idx])
        N = 10*nmp.log10(P[n_idx])
        print 'SNR: ', int(S-N), 'dB'
        if S-N > SNR_THRESH:
            return None
        else:
            print '...LOW SNR. QUITTING!'
            sys.exit()

    def recover_carrier(self, samples):
        print "RECOVER CARRIER"
        crr = sig.lfilter(self.filters.ipf[0], self.filters.ipf[1], samples)
        return sig.hilbert(crr) ** 3.0

    def recover_clock(self, samples):
        print "RECOVER CLOCK"
        clk = sig.lfilter(self.filters.clk[0], self.filters.clk[1], samples)
        clk = nmp.array(clk > 0)
        return clk - 0.5

    def recover_symbols(self, clk, x):
        print "RECOVER SYMBOLS"
        zero_xing = nmp.where(nmp.diff(nmp.sign(clk)))[0]
        zero_xing = zero_xing[0::2]
        amp = nmp.absolute(x[zero_xing])
        phz = nmp.angle(x[zero_xing])
        x = x[zero_xing]
        sym = (phz > 0)
        I = amp * nmp.cos(phz)
        Q = amp * nmp.sin(phz)
        sym = (Q>0)

        #self.plot.time_domain(amp, phz, sym)
        #self.plot.constellation(I, Q)
        return sym

    def synchronize(self, bits):
        print "SYNCHRONIZE BITS"
        words = []
        blocks = []
        groups = []
        m = 0

        for n, bit in enumerate(bits[:-26]):
            syndrome = self.code.syndrome(bits, n)
            # [984, 980, 604, 600]
            if syndrome in self.code.syndromes:
                i = self.code.syndromes.index(syndrome)     # 1,2,3,4 = A,B,C,D
                words.append([i, n, n - m])   # [letter, location, distance from prev word]
                m = n

        for n, word in enumerate(words):
            if (word[0] - words[n-1][0] == 1 and word[2] == 26) or word[0] == 0:
                blocks.append(word)

        for n, block in enumerate(blocks[:-3]):
            group = [word[0] for word in blocks[n:n+4]]
            if nmp.array_equal(group, self.code.words):
                groups.append(block[1])

        print "...WORDS: ", len(words)
        print "...BLOCKS:", len(blocks)
        print "...GROUPS:", len(groups)

        for group in groups:
            self.decode(bits[group:group + 104])

    def decode(self, bits):
        A = 1 * bits[0:16]
        B = 1 * bits[26:42]
        C = 1 * bits[52:68]
        D = 1 * bits[78:94]

        self.code.prog_id(A)
        self.code.group_type(B)
        self.code.prog_type(B)

        if self.code.gt == '0A':
            self.code.prog_service(B, D)
        elif self.code.gt == '2A':
            self.code.radiotext(B, C, D)

def main():
    filt = Filters(F_SAMPLE, F_SYM, DEC_RATE)
    code = Code()
    plot = Plot()
    rds = Radio_Data_System(filt, code, plot)

    rds.start()
    code.print_code()

LIVE = False

station = float(sys.argv[1]) * 1e6
F_CENTER = station     # FM Station
F_SAMPLE = 228e3       # 225-300 kHz and 900 - 2032 kHz
N_SAMPLES = 512*512*3  # must be multiple of 512, should be a multiple of 16384 (URB size)
DEC_RATE = 12          # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
F_SYM = 1187.5            # Symbol rate, full bi-phase
SPS = F_SAMPLE/DEC_RATE/F_SYM   # Samples per Symbol
SNR_THRESH = 5          #SNR threshold for quitting, dB

main()
