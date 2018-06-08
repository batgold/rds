#!/usr/bin/env python
import numpy as nmp
import scipy.signal as sig
import decode
import graph
import constants
import filters

def demod_fm(x):
    x = x[1:] * nmp.conj(x[:-1])
    fm = nmp.angle(x)
    return fm

def receive(que):
    g = graph.Graph()
    while True:
        data = que.get(timeout=5)
        if data is None:
            break
        demod_rds(data, g)
        #demod_old(data, g)

def demod_rds(x, g):
    bpf_57k = filters.bpf_57k()
    peak_19k = filters.peak_19k()
    lpf = filters.lpf()
    clk = filters.clk()

    x_bpf = sig.lfilter(bpf_57k[0], bpf_57k[1], x) # BPF
    pilot_19 = sig.filtfilt(peak_19k[0], peak_19k[1], x)
    pilot_57 = pilot_19**3
    bb = x_bpf*pilot_57
    bb_lpf = 1.2e3*sig.lfilter(lpf[0], lpf[1], bb)
    bb_dec = bb_lpf[::constants.rds_dec]
    #bb_dec = sig.decimate(
            #bb_lpf, q=constants.rds_dec, ftype='iir', zero_phase=True)

    fsym = constants.fsym/(constants.fs/constants.rds_dec)
    bb_i = bb_dec * nmp.cos(2*nmp.pi*2*fsym*nmp.arange(len(bb_dec)))
    bb_q = bb_dec * nmp.sin(2*nmp.pi*2*fsym*nmp.arange(len(bb_dec)))

    #bb_phz = nmp.arctan2(bb_q, bb_i)
    rate = int(constants.fs/constants.rds_dec/constants.fsym)

    #clk_tone = sig.lfilter(clk[0], clk[1], bb_dec)
    #clk = (clk_tone > 0)
    #clk = clk - 0.5

    symbols = (bb_i > 0)
    symbols = symbols[::rate]
    #symbols = nmp.abs(bb_phz) < nmp.pi/2
    #bb_i = bb_i[::rate]
    #bb_q = bb_q[::rate]
    #symbols = (bb_i > 0)
    bits = nmp.bitwise_xor(symbols[1:], symbols[:-1])

    g.scope(x_bpf, bb_lpf, bb_dec, symbols, bb_i)
    g.spectrum(x)
    g.spectrum2(bb_dec)
    g.constellation(bb_i, bb_q, rate)
    g.run()

    decode.group_sync(bits)

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

