#!/usr/bin/env python

import sys
import pickle
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.integrate import cumtrapz
from filters import Filters
from graph import Graph
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

class Demodulate():
    """RADIO"""

    def __init__(self):
        self.bits = []
        self.start()

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
        #graph.fm(x2)
        self.calc_snr(x2)
        crr = self.recover_carrier(x2)  # get 57kHz carrier
        x3 = sig.lfilter(filters.bpf[0], filters.bpf[1], x2) # BPF
        x4 = 1000 * crr * x3        # mix down to baseband
        x5 = sig.decimate(x4, DEC_RATE, zero_phase=True)
        #graph.bb(x5)
        clk = self.recover_clock(x5)
        x6 = 90 * sig.lfilter(filters.rrc, 1, x5)
        sym = self.recover_symbols(clk, x6)
        self.bits = nmp.bitwise_xor(sym[1:], sym[:-1])

    def calc_snr(self, x):
        [P, f] = plt.psd(x, NFFT=2048, Fs=F_SAMPLE)
        s_idx = (nmp.abs(f - (57e3 + F_SYM))).argmin()
        n_idx = (nmp.abs(f - 57e3)).argmin()
        S = 10*nmp.log10(P[s_idx])
        N = 10*nmp.log10(P[n_idx])
        print 'SNR: ', int(S-N), 'dB'
        if S-N < SNR_THRESH:
            print '...LOW SNR. QUITTING!'
            sys.exit()

    def recover_carrier(self, samples):
        print "RECOVER CARRIER"
        crr = sig.lfilter(filters.ipf[0], filters.ipf[1], samples)
        return sig.hilbert(crr) ** 3.0

    def recover_clock(self, samples):
        print "RECOVER CLOCK"
        clk = sig.lfilter(filters.clk[0], filters.clk[1], samples)
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

        #graph.time_domain(amp, phz, sym)
        #graph.constellation(I, Q)
        return sym

class Decode():
    """DECODE"""
    def __init__(self):
        self.bits = []
        self.n = 0

    def start(self):
        self.block_sync()

    def block_sync(self):
        while self.n < len(self.bits - 104):
            syns = self.group_syndrome()
            if syns == code.syndromes:
                print "OK", self.n
                self.decode()
                self.n += 104
            else:
                self.n = self.error_correct()
            self.n += 1

    def error_correct(self):
        return self.n
        #syndrome error
        err = 


        #shift right

    def process_bits2(self, n):
        print "PROCESS BITS"
        group_num = 0

        for n in range(bit_sync,len(self.bits),26):
            syndrome = code.syndrome(self.bits[n:n+26])
            print syndrome
            print '-', syndrome ^ code.syndromes[group_num]
            err = syndrome ^ code.syndromes[group_num]
            if err <> 0:
                self.error_correct(n, group_num, err)
        #self.decode(self.bits[n:n+104])
            group_num += 1
            if group_num == 4: group_num = 0

    def error_correct2(self, n, g, s):
        print "ERROR"
        syndrome = code.syndrome(self.bits[n+1:n+1+26])
        #print '--', syndrome ^ code.syndromes[g]
        #try to shift bits right 1
        if syndrome ^ code.syndromes[g] == 0:
            print "FIXED!"
        if s <= 512:
            try:
                bit = code.parity.index(s)
                print ">>>>",s, bit
                self.bits[n+bit] = 1 - self.bits[n+bit]
                syndrome = code.syndrome(self.bits[n:n+26])
                if syndrome ^ code.syndromes[g] == 0:
                    print "FIXED!"
            except:
                return None

    def decode(self):
        bits = self.bits[self.n:self.n+104]
        A = 1 * bits[0:16]
        B = 1 * bits[26:42]
        C = 1 * bits[52:68]
        D = 1 * bits[78:94]

        code.prog_id(A)
        code.group_type(B)
        code.prog_type(B)

        if code.gt == '0A':
            code.prog_service(B, D)
        elif code.gt == '2A':
            code.radiotext(B, C, D)

    def group_syndrome(self):
        n = self.n
        A = self.syndrome(self.bits[n:n+26])
        B = self.syndrome(self.bits[n+26:n+52])
        C = self.syndrome(self.bits[n+52:n+78])
        D = self.syndrome(self.bits[n+78:n+104])
        return [A, B, C, D]

    def syndrome(self, bits):
        syn = 0
        for n, bit in enumerate(bits):
            if bit:
                syn = nmp.bitwise_xor(syn, code.parity[n])
        return syn

def main():
    demod = Demodulate()
    decode = Decode()

    decode.bits = demod.bits
    decode.start()
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

filters = Filters(F_SAMPLE, F_SYM, DEC_RATE)
graph = Graph(F_SAMPLE, N_SAMPLES)
code = Code()

main()
