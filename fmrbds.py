#!/usr/bin/env python

import pickle
import commpy as com
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig
from bitarray import bitarray
#from bitstring import BitArray
from rtlsdr import RtlSdr

F_SAMPLE = int(228e3)       # 225-300 kHz and 900 - 2032 kHz
F_CENTER = int(96.5e6)      # FM Station, need strong RDBS signal, SF (KOIT XMAS)
#F_CENTER = int(88.5e6)      # FM Station, need strong RDBS signal, SF GAME!!!
#F_CENTER = int(97.7e6)      # FM Station, need strong RDBS signal, PAL
F_PILOT = int(19e3)         # pilot tone, 19kHz from Fc
N_SAMPLES = int(512*512*2)  # must be multiple of 512, should be a multiple of 16384 (URB size)
DEC_RATE = int(12)          # RBDS rate 1187.5 Hz. Thus 228e3/1187.5/24 = 8 sample/sym
F_SYM = 1187.5            # Symbol rate, full bi-phase
sps = int(F_SAMPLE/DEC_RATE/F_SYM)   # Samples per Symbol


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
        w = F_SYM / (float(F_SAMPLE) / float(DEC_RATE) / 2.0)
        q = w / 5.0 * F_SAMPLE * DEC_RATE        # Q = f/bw, BW = 5 Hz
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
        self.bb = []
        self.I = []
        self.Q = []
        self.sym = []
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
        self.crr = sig.hilbert(y1) ** 3.0                                # multiply up to 57 kHz
        #self.crr3 = nmp.exp(nmp.arange(0,len(self.data))*1j*F_PILOT*3)          # free-running clock

    def cull_rds(self):
        """STEP 2.3: FILTER, MIX DOWN TO BB & DECIMATE"""
        print("...MOVE TO BASEBAND")
        x2 = sig.lfilter(filter.bpf[0], filter.bpf[1], self.fm)     # BPF
        x3 = 1000 * x2 * self.crr                                 # mix signal down by 57 kHz
        self.bb = sig.decimate(x3, DEC_RATE, zero_phase=True)      # Decimate

    def pulse_shape(self):
        """STEP 2.4: APPLY R.R.COSINE FILTER"""
        print("...PULSE SHAPE")
        x4 = 10 * sig.lfilter(filter.rrc, 1, self.bb)     #gain of 5 seems gud
        self.bb2 = x4
        self.I = nmp.real(x4)
        self.Q = nmp.imag(x4)
        self.phz = nmp.arctan2(self.I, self.Q)

    def clock_recovery(self):
        """STEP 2.5: DETERMINE SYMBOL CLOCK RATE"""
        print("...RECOVER CLOCK")
        x6 = 1000 * sig.lfilter(filter.clk[0], filter.clk[1], self.bb)
        x6 = nmp.real(x6)
        self.clk = (x6 > 0)

    def symbol_decode(self):
        """STEP 2.6: SAMPLE SYMBOLS AT CLOCK RATE"""
        print("...DECODING SYMBOLS")
        symbol = 0
        n_sym = 0
        n = 1
        m = 1
        while n < len(self.bb):
            if self.clk[n] == 1 and self.clk[n-1] == 0:       #capture signal at clock ON
                symbol = 0
                m = 0
                n_sym += 1
                while self.clk[m] == 1:
                    symbol = symbol + self.bb2[n + m]
                    m += 1
            self.sym = symbol
            n += 1

        x5 = self.phz[self.phz_offset::sps]
        self.I = self.I[self.phz_offset::sps]
        self.Q = self.Q[self.phz_offset::sps]
        self.sym = (x5 > 0)

    def bit_decode(self):
        """STEP 2.7: DECODE MANCHESTER SYMBOLS"""
        self.bits = nmp.bitwise_xor(self.sym[1:], self.sym[:-1])


class Block_Sync:
    """STEP 3: Synchronize and Interrogate Blocks"""

    def __init__(self, data):
        print("SYNCHRONIZING BLOCKS")
        self.data = data
        self.offset = []
        self.blocks = []
        self.parity = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
        #self.syndromes = [984, 980, 604, 972, 600]
        self.syndromes = [984, 980, 604, 600]
        #self.offsets = ['A', 'B', 'C', 'Cp', 'D']
        self.offsets = ['A', 'B', 'C', 'D']
        self.get_offsets()
        print(len(self.offset))

    def get_offsets(self):
        last_bit = 0
        for n_bit in range(len(self.data)-26):          #for each bit in data
            syndrome = self.get_syndrome(n_bit)         #calculate syndrome

            if syndrome in self.syndromes:
                indx = self.syndromes.index(syndrome)
                self.offset.extend(self.offsets[indx])
                #print self.offsets[s_indx], n_bit - last_bit, n_bit
                last_bit = n_bit

    def verify_blocks(self):
        blocks = self.offset.index(self.offsets)

    def get_syndrome(self, n_bit):
        syndrome = 0;
        for n in range(26):
            if self.data[n_bit + n]:
                syndrome = nmp.bitwise_xor(syndrome, self.parity[n])    #multiply word by H
        return syndrome


class Decode:
    """STEP 4: Decode Bits into Message"""

    def __init__(self, data, bit):
        self.K = 4096
        self.W = 21672
        self.data = data
        self.bit = bit
        self.A = 1 * data[bit:bit + 16]
        self.B = 1 * data[bit + 26:bit + 42]
        self.C = 1 * data[bit + 53:bit + 69]
        self.D = 1 * data[bit + 79:bit + 95]
        self.pi = []
        self.gt = []
        self.pt = []
        self.prog_id()
        self.group_type()
        self.prog_type()

    def prog_id(self):
        """Program Identification"""
        pi_int = self.bit2int(self.A)
        if pi_int > self.W:
            return None
        else:
            pi1 = self.K
            pi2 = nmp.floor_divide(pi_int - pi1, 676)
            pi3 = nmp.floor_divide(pi_int - pi1 - pi2*676, 26)
            pi4 = pi_int - pi1 - pi2*676 - pi3*26
            self.pi = 'K' + chr(pi2+65) + chr(pi3+65) + chr(pi4+65)

    def group_type(self):
        """Group Type"""
        gt_num = self.bit2int(self.B[0:4])
        gt_ver = self.B[4]
        self.gt = str(gt_num) + chr(gt_ver + 65)      # 5th bit = Version (A|B)

    def prog_type(self):
        """Program Type"""
        pt_num = self.bit2int(self.B[6:11])
        self.pt = self.pty_dict(pt_num)

    def char_code(self):
        """Character Placement"""
        B = self.B[:-2]

    def bit2int(self, bits):
        """Convert bit string to integer"""
        word = 0
        for bit in bits:                # First convert to bitarray
            word = (word<<1) | bit
        return int(word)

    def pty_dict(self, code):
        """"Program Type Codes"""
        pty_codes = ['None', 'News', 'Info', 'Sports', 'Talk', 'Rock', \
                    'Classic Rock', 'Adult Hits', 'Adult Hits', 'Soft Rock', \
                    'Top 40', 'Country', 'Oldies', 'Soft', 'Nostalgia', 'Jazz', \
                    'Classical', 'R&B', 'Foreign Lang.', 'Religious Music', \
                    'Religious Talk', 'Personality', 'Public', 'College', 'NA', \
                    'Weather', 'Emergency Test', 'ALERT!']
        return pty_codes[code]

live = False

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
I = data.I
Q = data.Q
bits = data.bits

blocks = Block_Sync(bits).blocks

msg = Decode(bits, 463)

prog_id = msg.pi
print(prog_id)
group_type = msg.gt
print(group_type)
prog_type = msg.pt
print(prog_type)


# PLOTS
#plt.figure()
#plt.plot(I, Q, 'b.', alpha=0.5)
#
plt.figure()
plt.plot(nmp.real((data.bb2[6000:6160])),'b.')
plt.plot((data.phz[6000:6160]),'r+')
plt.plot((data.clk[6000:6160]),'g')

#plt.psd(data.bb2,NFFT=2048,Fs=F_SAMPLE/DEC_RATE)


plt.show()
