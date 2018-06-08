#!/usr/bin/env python
import sys
import numpy as nmp
import scipy.signal as sig
import graph
import constants
import filters


def group_sync(bits):
    n = 0
    groups = []

    while n < len(bits) - 104:
        n += 1
        frame = bits[n:n+104]
        if group_syndrome(frame) == constants.syndromes:
            print 'sync'
            unpack_msg(frame)
            break

    for m in range(n, len(bits), 104):
        frame = bits[m:m+104]
        groups.append(frame)

    unpack_groups(groups)

def block_syndrome(bits):
    syn = 0
    for n, bit in enumerate(bits):
        if bit:
            syn = nmp.bitwise_xor(syn, constants.parity[n])
    return syn

def group_syndrome(frame):
    syn = {}
    syn['A'] = block_syndrome(frame[:26])
    syn['B'] = block_syndrome(frame[26:52])
    syn['C'] = block_syndrome(frame[52:78])
    syn['D'] = block_syndrome(frame[78:])
    return syn

def unpack_groups(groups):
    for group in groups:
        unpack_msg(group)

def demod_old(x, g):
    bpf_57k = filters.bpf_57k()
    clk = filters.clk()
    rrc = filters.rrc()

    x3 = sig.lfilter(bpf_57k[0], bpf_57k[1], x)
    crr = recover_carrier(x)  # get 57kHz carrier
    x4 = crr * x3        # mix down to baseband
    x5 = sig.decimate(x4, constants.rds_dec, zero_phase=True)
    x6 = 100*sig.lfilter(rrc[0], rrc[1], x5)
    clk = recover_clock(x5)
    sym = recover_symbols(clk, x6)
    bits = nmp.bitwise_xor(sym[1:], sym[:-1])

    #g.scope(x3, crr, x6)  #, symbols, bb_i)
    #g.spectrum(x)
    #g.spectrum2(x5)
    #g.constellation(sym, sym, 16)
    #g.run()

    group_sync(bits)

def demod2(x, g):
    x1 = samples[1:] * nmp.conj(samples[:-1])    # 1:end, 0:end-1
    x2 = nmp.angle(x1)
    x3 = sig.lfilter(self.filters.bpf[0], self.filters.bpf[1], x2) # BPF
    crr = self.recover_carrier2(x2)  # get 57kHz carrier
    x4 = crr * x3        # mix down to baseband
    x5 = sig.decimate(x4, self.dec_rate, zero_phase=True)
    x6 = 0.3*sig.lfilter(self.filters.rrc, 1, x5)
    clk = 0.5*self.recover_clock(x5)
    sym = self.recover_symbols(clk, x6)
    self.bits = nmp.bitwise_xor(sym[1:], sym[:-1])
    self.decode()
    snr = self.calc_snr(x2)
    self.graph.update(x2, x3, x4, phz, x6, clk, self.pi_sync, self.pt_sync, \
        self.ps, self.rt, self.valid_group_cnt, self.group_cnt, snr)
    #self.print_code()

def recover_carrier(x):
    peak_19k = filters.peak_19k()
    y1 = sig.lfilter(peak_19k[0], peak_19k[1], x)
    y2 = sig.hilbert(y1) ** 3.0
    return y2

def recover_carrier2(self, x):
    t = nmp.arange(0,len(x))
    return nmp.exp(1j*2*nmp.pi*Fc/self.Fs*t)

def recover_clock(x):
    clk = filters.clk()
    y0 = sig.lfilter(clk[0], clk[1], x)
    y1 = nmp.array(y0 > 0)
    return y1 - 0.5

def recover_symbols(clk, x):
    zero_xing = nmp.where(nmp.diff(nmp.sign(clk)))[0]
    #zero_xing = zero_xing[::2]
    zero_xing = zero_xing[0:-70:2]
    #y = x[zero_xing]
    y = x[zero_xing+25] #25, 31, 66
    amp = nmp.absolute(y)
    phz = nmp.angle(y)
    I = amp * nmp.cos(phz)
    Q = amp * nmp.sin(phz)
    #sym = (I > 0)
    sym = (Q > 0)
    return sym

def unpack_msg(frame):
    A = frame[:16]
    B = frame[26:42]
    C = frame[52:68]
    D = frame[78:94]

    pi = prog_id(A)
    pt = prog_type(B)
    gt = group_type(B)
    print pi, pt, gt

    if gt == '0A':
        ps = prog_service(B, D)
    elif gt == '2A':
        rt = radiotext(B, C, D)
        print rt

def sync_check(self):
    if self.pi == self.pi_sync:
        return True

def prog_id(A):
    """Program Identification"""
    A = bit2int(A)
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

def prog_type(B):
    """Program Type"""
    pt_num = bit2int(B[6:11])
    try:
        return constants.pt_codes[pt_num]
    except:
        return 'ERR'

def group_type(B):
    """Group Type"""
    gt_num = bit2int(B[0:4])
    gt_ver = B[4]
    return str(gt_num) + chr(gt_ver + 65)      # 5th bit = Version (A|B)

def prog_service(B, D):
    ps = ['_']*8
    pschr = [0,0]
    cc = bit2int(B[-2:]) - 1
    pschr[0] = bit2int(D[0:8])
    pschr[1] = bit2int(D[8:16])
    ps[2*cc:2*cc+2] = [chr(pschr[i]) if 32 < pschr[i] < 128 else '_' for i in range(0,2)]
    return ps

def radiotext(B, C, D):
    rt = ['_']*64
    rtchr = [0,0,0,0]
    cc = bit2int(B[-4:])
    rtchr[0] = bit2int(C[0:8])
    rtchr[1] = bit2int(C[8:16])
    rtchr[2] = bit2int(D[0:8])
    rtchr[3] = bit2int(D[8:16])
    rt[4*cc:4*cc+4] = [chr(rtchr[i]) if 32 < rtchr[i] < 128 else '_' for i in range(0,4)]
    return rt

def bit2int(bits):
    """Convert bit string to integer"""
    word = 0
    for bit in bits:
        word = (word<<1) | bit
    return int(word)

def init():
    self.pi = ''
    self.pt = ''
    self.gt = ''
    self.ps = ['_']*8
    self.rt = ['_']*64
    self.bits = bits

def calc_snr(x):
    F, Y = sig.welch(x, fs=fs, nfft=2048, return_onesided=False)
    s_idx = (nmp.abs(F - (57e3 + fsym))).argmin()
    n_idx = (nmp.abs(F - 57e3)).argmin()
    S = 20*nmp.log10(Y[s_idx])
    N = 20*nmp.log10(Y[n_idx])
    return S-N

def costas(x):
    n = len(x)
    k = 0.003
    y_cos = nmp.zeros(n)
    y_sin = nmp.zeros(n)
    phz = nmp.zeros(n)
    #phz[0] = phz_offset
    phz[0] = 0
    for n, xn in enumerate(x[:-1]):
        y_cos[n] = 2*xn*nmp.cos(2*nmp.pi*fc/fs*n + phz[n])
        y_sin[n] = 2*xn*nmp.sin(2*nmp.pi*fc/fs*n + phz[n])
        phz[n+1] = phz[n] - k*y_cos[n]*y_sin[n]

    phz_offset = phz[-1]
    return y_cos, phz

def costas3(self, x):
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

