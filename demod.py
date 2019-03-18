#!/usr/bin/env python
import graph
import decode
import filters
import numpy as np
import constants as co
import scipy.signal as sg

def receive(que):
    #TODO: move to fmrx
    g = graph.Graph()
    rds = decode.RDS()      #<<< this is called only once
    while True:
        data = que.get(timeout=5)
        if data is None:
            break

        g.clf()
        bits = demod(data, g)
        decode.decode(bits, g, rds)
        g.run()

def demod(fm, g):

    bb = fm2bb(fm, g)
    bits = bb2bit(bb, g)

    return bits

def fm2bb(x, g):
    """Convert FM signal to Baseband"""
    bpf_57k = filters.bpf_57k()
    peak_19k = filters.peak_19k()
    lpf = filters.lpf()

    # Get Carrier to Demodulate
    x_19 = sg.filtfilt(peak_19k[0], peak_19k[1], x)
    x_57 = x_19**3
    #TODO BPF the CARRIER 57kHz
    #x_57 = sg.lfilter(bpf_57k[0], bpf_57k[1], x_57)

    # Demodulate, Filter, & Decimate
    x_bpf = sg.lfilter(bpf_57k[0], bpf_57k[1], x)
    x_bb = x_bpf * x_57
    x_bb_lpf = 1.3e3*sg.lfilter(lpf[0], lpf[1], x_bb)
    #bb_rrc = 100*sg.lfilter(rrc[0], rrc[1], bb_dec)
    x_bb_dec = x_bb_lpf[0::co.rds_dec]

    freq, fm = get_spec(x, 512)
    snr = calc_snr(fm, 512)

    g.spectrum(freq, fm, snr)
    g.scope(x_bpf=x_bpf, bb_lpf=x_bb_lpf)

    return x_bb_dec

def bb2bit(x, g):
    """Convert Baseband Signal to Bits"""
    sym_len = int(co.fs/co.fsym/co.rds_dec)
    sym_cnt = int(len(x)/sym_len)
    sym_list = [x[n*sym_len:n*sym_len+sym_len] for n in range(sym_cnt)]

    T = sym_len/2
    for sym in sym_list:
        T = sym_detect(T, np.abs(sym), sym_len/2)

    g.eye(sym_list, sym_len, T)

    bit_list = []
    for sym in sym_list:
        bit_list.append(sym[T])

    #bit_list = [sym[16::32] for sym in sym_list]
    bit_slice = [1 if i>0 else 0 for i in bit_list]
    bits = np.bitwise_xor(bit_slice[1:], bit_slice[:-1])
    return bits

def sym_detect(T, sym, sym_len):
    """Early-Late Detection"""
    d = 2   # small aperture

    if T + d >= sym_len:
        T = d + 1
    if T - d < 1:
        T = sym_len - d - 1

    if sym[T-d] > sym[T+d]:
        T -= d/2
    elif sym[T-d] < sym[T+d]:
        T += d/2

    return int(T)

def get_spec(x, n):
    F, X = sg.welch(x, fs=co.fs, nfft=n, return_onesided=True)
    XdB = 20 * np.log10(X)
    return F, XdB

def calc_snr(x, n):
    s_idx = int((co.fc+co.fsym)/co.fs*n)
    n_idx = int((co.fc+3*co.fsym)/co.fs*n)  # just outside rds bw
    S = x[s_idx]
    N = x[n_idx]
    return S - N

def recover_carrier(x):
    peak_19k = filters.peak_19k()
    y1 = sg.lfilter(peak_19k[0], peak_19k[1], x)
    y2 = sg.hilbert(y1) ** 3.0
    return y2

def recover_clock(x):
    clk = filters.clk()
    y0 = sg.lfilter(clk[0], clk[1], x)
    y1 = np.array(y0 > 0)
    return y1 - 0.5

def recover_symbols(clk, x):
    zero_xing = np.where(np.diff(np.sign(clk)))[0]
    #zero_xing = zero_xing[::2]
    zero_xing = zero_xing[0:-70:2]
    #y = x[zero_xing]
    y = x[zero_xing+25] #25, 31, 66
    amp = np.absolute(y)
    phz = np.angle(y)
    I = amp * np.cos(phz)
    Q = amp * np.sin(phz)
    #sym = (I > 0)
    sym = (Q > 0)
    return sym

