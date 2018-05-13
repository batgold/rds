#!/usr/bin/env python
import sys
import time
import curses
import numpy as nmp
import scipy.signal as sig
from matplotlib.pyplot import psd
import matplotlib.pyplot as plt
from filters import Filters
from graph import Graph
from rtlsdr import RtlSdr

class RDS():
    """RDS"""
    def __init__(self, stdscr, Fs, Ns, station):
        self.stdscr = stdscr
        self.Fs = Fs
        self.Ns = Ns
        self.station = station
        self.Fc = 57e3              # Center Frequency of RDS, 57kHz
        self.Fsym = 1187.5            # Symbol rate, full bi-phase
        #self.dec_rate = 12          # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
        self.dec_rate = 6          # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
        self.snr_min = 1          #SNR threshold for quitting, dB
        self.phz_offset = 0
        self.n = 0
        self.bits = []
        self.sync = False
        self.parity = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
        self.syndromes = {'A':984, 'B':980, 'C':604, 'D':600}
        self.pi = ''
        self.ps = list('________')
        self.rt = list('________________________________________________________________')
        self.pt = ''
        self.gt = ''
        self.group_cnt = 0
        self.valid_group_cnt = 0

        self.graph = Graph(self.Fs, self.Ns, self.station)
        self.filters = Filters(self.Fs, self.Fsym, self.dec_rate)

        self.bpf = self.filters.bpf
        #curses.use_default_colors()

    def demod(self, samples):
        x1 = samples[1:] * nmp.conj(samples[:-1])    # 1:end, 0:end-1
        x2 = nmp.angle(x1)
        #self.calc_snr(x2)
        #crr = self.recover_carrier(x2)  # get 57kHz carrier
        x3 = sig.lfilter(self.bpf[0], self.bpf[1], x2) # BPF
        x4, x4b, phz = self.costas2(x3)
        #x4 = x4 * nmp.cos(nmp.arange(0,len(x4))*2*nmp.pi*57e3/self.Fs)
        #x3 = sig.lfilter(self.filters.bpf[0], self.filters.bpf[1], x2) # BPF
        #x4 = crr * x2        # mix down to baseband
        x5 = sig.decimate(x4, self.dec_rate, zero_phase=True) * 1000
        x6 = sig.lfilter(self.filters.rrc, 1, x5)
        clk = self.recover_clock(x5)
        sym = self.recover_symbols(clk, x6)
        self.bits = nmp.bitwise_xor(sym[1:], sym[:-1])
        #self.decode()
        self.graph_all(x2, x3, x4, x4b, phz, x6)
        #self.print_code()

    def graph_all(self, x2, x3, x4, x4b, phz, x6):
        self.graph.fm = x2
        self.graph.fm_bpf = x3
        self.graph.bb = x4
        self.graph.bb2 = x4b
        self.graph.bb3 = x6
        #self.graph.sym = sym
        #self.graph.clk = clk
        self.graph.phz_offset = phz
        #self.graph.pt = self.pt
        #self.graph.pi = self.pi
        #self.graph.ps = ''.join(self.ps)
        #self.graph.rt = ''.join(self.rt)
        self.graph.update()

    def calc_snr(self, x):
        #USE SIG.WELCH!!!
        [P, f] = psd(x, NFFT=2048, Fs=self.Fs)
        s_idx = (nmp.abs(f - (57e3 + self.Fsym))).argmin()
        n_idx = (nmp.abs(f - 57e3)).argmin()
        S = 10*nmp.log10(P[s_idx])
        N = 10*nmp.log10(P[n_idx])
        #self.stdscr.addstr(0,0,'SNR: {0}dB'.format(int(S-N)))
        #self.stdscr.refresh()
        if S-N < self.snr_min:
            ##self.stdscr.addstr(1,0,'...LOW SNR. QUITTING!')
            #self.stdscr.refresh()
            #self.graph.fm(x)
            #sys.exit()
            return None

    def costas(self, x):
        print 'costas start'
        n = len(x)
        N = self.filters.costas_bw
        fc = 57e3/self.Fs/2
        k = 0.003
        y1 = nmp.zeros(n)
        y2 = nmp.zeros(n)
        phz = nmp.zeros(n)
        z_cos = nmp.zeros(N)
        z_sin = nmp.zeros(N)
        phz[0] = self.phz_offset
        h = self.filters.costas_lpf
        for n, xn in enumerate(x[:-1]):
            z_cos[:-1] = z_cos[1:]
            z_cos[-1] = 2*xn*nmp.cos(2*nmp.pi*fc*n + phz[n])
            z_sin[:-1] = z_sin[1:]
            z_sin[-1] = 2*xn*nmp.sin(2*nmp.pi*fc*n + phz[n])
            lpf_cos = nmp.matmul(h, nmp.transpose(z_cos))
            lpf_sin = nmp.matmul(h, nmp.transpose(z_sin))
            phz[n+1] = phz[n] - k*lpf_cos*lpf_sin
            y1[n] = lpf_cos
            y2[n] = lpf_sin

        print 'costas end'
        self.phz_offset = phz[-1]
        return y1, y2, phz

    def costas2(self, x):
        n = len(x)
        phz = nmp.zeros(n)
        cos = nmp.zeros(n)
        sin = nmp.zeros(n)
        y1 = nmp.zeros(n)
        y2 = nmp.zeros(n)
        k = 8e-5
        bw = int(1/(self.Fc/self.Fs)*1)
        for n, xn in enumerate(x[:-1]):
            cos[n] = xn * nmp.cos(2*nmp.pi*xn*self.Fc/self.Fs + phz[n])
            sin[n] = xn * nmp.sin(2*nmp.pi*xn*self.Fc/self.Fs + phz[n])
            y1[n] = nmp.sum(cos[max(0,n-bw):n])
            y2[n] = nmp.sum(sin[max(0,n-bw):n])
            phz[n+1] = phz[n] - k*nmp.pi*nmp.sign(y1[n]*y2[n])
        return y1, y2, phz

    def recover_carrier(self, x):
        y1 = sig.lfilter(self.filters.ipf[0], self.filters.ipf[1], x)
        y2 = sig.hilbert(y1) ** 3.0
        return y2 * 1e2

    def recover_clock(self, x):
        y0 = sig.lfilter(self.filters.clk[0], self.filters.clk[1], x)
        y1 = nmp.array(y0 > 0)
        return y1 - 0.5

    def recover_symbols(self, clk, x):
        zero_xing = nmp.where(nmp.diff(nmp.sign(clk)))[0]
        #zero_xing = zero_xing[::2]
        z1 = zero_xing[:-1:2]
        z2 = zero_xing[1::2]
        #y = x[zero_xing]
        #amp = nmp.absolute(y)
        #phz = nmp.angle(y)
        #I = amp * nmp.cos(phz)
        #Q = amp * nmp.sin(phz)
        I = x[z1]
        Q = x[z2]
        sym = (I > 0)
        #self.graph.phz = phz
        #self.graph.amp = amp
        self.graph.I = I
        self.graph.Q = Q
        return sym

    def decode(self):
        self.n = 0
        self.sync = False

        while self.n < len(self.bits) - 104:
            if not self.sync:
                self.n += 1
                if self.group_syndrome() == self.syndromes:
                    self.set_anchor()

            if self.sync:
                bit_frame = self.bits[self.n:self.n + 104]
                self.unpack_msg(bit_frame)
                if self.sync_check():
                    self.print_code()
                    self.valid_group_cnt += 1
                self.n += 104
                self.group_cnt += 1

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

    def set_anchor(self):
        bit_frame = self.bits[self.n:self.n + 104]
        self.unpack_msg(bit_frame)
        self.pi_sync = self.pi
        self.sync = True

    def unpack_msg(self, bits):
        A = bits[:16]
        B = bits[26:42]
        C = bits[52:68]
        D = bits[78:94]

        self.pi = self.prog_id(A)
        self.pt = self.prog_type(B)
        self.gt = self.group_type(B)

        if self.gt == '0A':
            self.prog_service(B, D)
        elif self.gt == '2A':
            self.radiotext(B, C, D)

    def sync_check(self):
        if self.pi == self.pi_sync:
            return True

    def print_code(self):
        #self.stdscr.clear()
        head = self.pi + self.pt + self.gt
        ##self.stdscr.addstr(1, 0, "{0}".format(head))
        #self.stdscr.addstr(2, 0, "{0}".format(''.join(self.ps)))
        #self.stdscr.addstr(3, 0, "{0}".format(''.join(self.rt)))
        #self.stdscr.addstr(5, 0, "{0} / {1}".format(
            #self.valid_group_cnt, self.group_cnt))
        #self.stdscr.addstr(6, 0, "{0}".format(''))
        #time.sleep(0.5)
        #self.stdscr.refresh()

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
        pschr = [0,0]
        cc = self.bit2int(B[-2:]) - 1
        pschr[0] = self.bit2int(D[0:8])
        pschr[1] = self.bit2int(D[8:16])
        self.ps[2*cc:2*cc+2] = [chr(pschr[i]) for i in range(0,2)]

    def radiotext(self, B, C, D):
        rtchr = [0,0,0,0]
        cc = self.bit2int(B[-4:]) - 1
        rtchr[0] = self.bit2int(C[0:8])
        rtchr[1] = self.bit2int(C[8:16])
        rtchr[2] = self.bit2int(D[0:8])
        rtchr[3] = self.bit2int(D[8:16])
        self.rt[4*cc:4*cc+4] = [chr(rtchr[i]) if 32 >= rtchr[i] < 128 else '_' for i in range(0,4)]

    def bit2int(self, bits):
        """Convert bit string to integer"""
        word = 0
        for bit in bits:
            word = (word<<1) | bit
        return int(word)
