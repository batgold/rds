#!/usr/bin/python
import numpy as nmp
import matplotlib.pyplot as plt
import scipy.signal as sig

pi = nmp.pi
t = nmp.arange(0,1e4,dtype=int)
fs = 4096.0
fc = 100.0
fm = 15.0
alfa = 0.01
beta = alfa/1000
phz1 = 1.1

e_fc = nmp.exp(-1j*2*pi*fc/fs*t)

m = nmp.sin(2*pi*fm/fs*t)
c = nmp.cos(2*pi*fc/fs*t + phz1)
s = m*c
s_ = -1j*nmp.sign(s)
s_plus = s + s_
#m_rx = s_plus * nmp.exp(-1j*2*pi*fc/fs*t)

phz2 = nmp.zeros(len(t))
for n in t[:-1]:
    lo = e_fc[n] * nmp.exp(-1j*phz2[n])
    c_re = nmp.real(s_plus * lo)
    c_im = nmp.imag(s_plus * lo)
    q = c_re * c_im

    phz2[n+1] = beta*phz2[n]


    #r[n+1] = beta*qn + r[n]
    #p[n+1] = p[n] + fc*2*pi*n + alfa*qn + r[n]

#o = nmp.exp(-1j*p)

m_rx = e_fc * nmp.exp(-1j*phz2)
F,Y = sig.welch(m_rx, fs=fs, nfft=2048*4, return_onesided=False, nperseg=2*1024)
#plt.semilogy(F,Y)

N = 8
Wn = float(fm*1.0/(fs/2))
b, a = sig.butter(N=N, Wn=Wn, btype='low', analog=False)
y = -sig.lfilter(b, a, m_rx)
#w, h = sig.freqz(b, a)
#plt.semilogy(w*fs/2/pi, abs(h))

plt.plot(c)
plt.plot(m_rx)
plt.show()

