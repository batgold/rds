#!/usr/bin/env python

import sys
import pickle
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig
from filters import Filters
from graph import Graph
from rtlsdr import RtlSdr


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
        self.sync = False
        self.parity = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
        #self.syndromes = [984, 980, 604, 972, 600]
        self.syndromes = {'A':984, 'B':980, 'C':604, 'D':600}

    def start(self):
        while self.n < len(self.bits - 104):

            if not self.sync:
                self.n += 1
                if self.validate_syndrome():
                    self.set_anchor()

            if self.sync:
                bit_frame = self.bits[self.n:self.n + 104]
                msg = Message(bit_frame)
                print self.n
                msg.decode()
                msg.print_code()
                #self.sync_check()
                self.n += 104

            if self.n > 1000: break

    def sync_check(self):
        pi = self.msg['pi']
        pt = self.msg['pt']
        print pi, pt, self.pi_sync
        if pi == self.pi_sync and pt == self.pt_sync:
            self.sync = True
            self.print_code()
        else:
            self.sync = False
            self.error_correct()

    def error_correct(self):
        if not self.sync:
            self.pi_bit_flip()
        if not self.sync:
            self.bit_slip()
        if not self.sync:
            a=1

        if self.sync:
            self.print_code()

    def bit_slip(self):
        n = self.n
        for x in range(-1,1,2):
            self.n += x
            self.decode()
            pi = self.msg['pi']
            if pi == self.pi_sync:
                self.sync = True
        self.n = n

    def pi_bit_flip(self):
        pi = self.msg['pi']
        pi = [ord(char) for char in pi]
        pi_sync = [ord(char) for char in self.pi_sync]
        pi_cnt = set(pi).intersection(pi_sync)
        if len(pi_cnt) >= 3:
            self.sync = True

    def bit_flip(self):
        s = self.check_syndrome()
        err = s['A'] ^ self.syndromes['A']
        err_size = bin(err).count('1')
        if err_size > 0 and err_size < 3:
            print err ^ s['A']
            self.n += 103
            self.sync = False
        else:
            self.sync = False

    def set_anchor(self):
        bit_frame = self.bits[self.n:self.n + 104]
        self.msg = Message(bit_frame)
        self.msg.decode()
        self.pi_sync = self.msg.pi
        self.sync = True

    def validate_syndrome(self):
        syn = self.group_syndrome()
        if syn == self.syndromes:
            return True

    def group_syndrome(self):
        bit_frame = self.bits[self.n:self.n + 104]
        syn = {}
        syn['A'] = self.block_syndrome(bit_frame[:26])
        syn['B'] = self.block_syndrome(bit_frame[26:52])
        syn['C'] = self.block_syndrome(bit_frame[52:78])
        syn['D'] = self.block_syndrome(bit_frame[78:])

        return syn

    def block_syndrome(self, bits):
        syn = 0
        for n, bit in enumerate(bits):
            if bit:
                syn = nmp.bitwise_xor(syn, self.parity[n])
        return syn

class Message():
    def __init__(self, bits):
        self.bits = bits
        self.pi = ''
        self.ps = list('________')
        self.rt = list('________________________________________________________________')
        self.pt = ''
        self.gt = ''

    def decode(self):
        A = self.bits[:16]
        B = self.bits[26:42]
        C = self.bits[52:68]
        D = self.bits[78:94]

        self.pi = self.prog_id(A)
        self.pt = self.prog_type(B)
        self.gt = self.group_type(B)

        if self.gt == '0A':
            self.prog_service(B, D)
            self.rt = ''
        elif self.gt == '2A':
            self.radiotext(B, C, D)
            self.ps = ''
        else:
            self.rt = ''
            self.ps = ''

    def print_code(self):
        print '\n', self.pi, self.pt, self.gt

        if self.gt == '0A':
            print ''.join(self.ps)
        elif self.gt == '2A':
            print ''.join(self.rt)
        else:
            print ''

    def prog_id(self, A):
        """Program Identification"""
        A = self.bit2int(A)
        K = 4096
        W = 21672
        if A >= W:
            pi0 = 'W'
            pi1 = W
        else:
            pi0 = 'K'
            pi1 = K
        pi2 = nmp.floor_divide(A - pi1, 676)
        pi3 = nmp.floor_divide(A - pi1 - pi2*676, 26)
        pi4 = A - pi1 - pi2*676 - pi3*26

        #self.pi = pi0 + chr(pi2+65) + chr(pi3+65) + chr(pi4+65)
        return pi0 + chr(pi2+65) + chr(pi3+65) + chr(pi4+65)

    def prog_type(self, B):
        """Program Type"""
        pt_codes = ['None', 'News', 'Info', 'Sports', 'Talk', 'Rock', \
                    'Classic Rock', 'Adult Hits', 'Soft Rock', \
                    'Top 40', 'Country', 'Oldies', 'Soft', 'Nostalgia', 'Jazz', \
                    'Classical', 'R&B', 'SoftR&B', 'Foreign Lang.', 'Religious Music', \
                    'Religious Talk', 'Personality', 'Public', 'College', 'NA', \
                    'Weather', 'Emergency Test', 'ALERT!']
        pt_num = self.bit2int(B[6:11])
        try:
            return pt_codes[pt_num]
        except:
            return 'ERR'

    def group_type(self, B):
        """Group Type"""
        gt_num = self.bit2int(B[0:4])
        gt_ver = B[4]
        return str(gt_num) + chr(gt_ver + 65)      # 5th bit = Version (A|B)

    def prog_service(self, B, D):
        cc = self.bit2int(B[-2:])
        ps1 = self.bit2int(D[0:8])
        ps2 = self.bit2int(D[8:16])
        self.ps[2*cc] = chr(ps1) + chr(ps2)

    def radiotext(self, B, C, D):
        rtchr = [0,0,0,0]
        cc = self.bit2int(B[-4:])
        rtchr[0] = self.bit2int(C[0:8])
        rtchr[1] = self.bit2int(C[8:16])
        rtchr[2] = self.bit2int(D[0:8])
        rtchr[3] = self.bit2int(D[8:16])

        #rt[4*cc:4*cc+4] = chr(rt1) + chr(rt2) + chr(rt3) + chr(rt4)
        self.rt[4*cc:4*cc+4] = [chr(rtchr[i]) for i in range(0,4)]

    def bit2int(self, bits):
        """Convert bit string to integer"""
        word = 0
        for bit in bits:
            word = (word<<1) | bit
        return int(word)

def main():
    demod = Demodulate()
    decode = Decode()

    decode.bits = demod.bits
    decode.start()


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

if __name__ == '__main__':
     main()
