#!/usr/bin/env python
import sys
from rtlsdr import RtlSdr
from fm_rds import RDS
from fm_audio import Audio

def callback(samples, sdr):
    global r
    r += 1
    if r > 3:
        sdr.cancel_read_async()
    else:
        #fm_audio.demod(samples)
        fm_rds.demod(samples)

def main():
    Fs = 228e3
    Ns = int(512*512)
    fm_rds = RDS(Fs=Fs, Ns=Ns, station)
    #fm_audio = Audio(Fs, Ns)

    sdr = RtlSdr()
    sdr.sample_rate = Fs
    sdr.center_freq = station
    sdr.gain = 'auto'
    sdr.read_samples_async(callback, Ns)
    sdr.close()
    #fm_audio.close()

if __name__ == "__main__":
    r = 0
    station = float(sys.argv[1]) * 1e6
    main()
