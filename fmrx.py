#!/usr/bin/env python

import sys
#import fm_rds
import numpy as nmp
from rtlsdr import RtlSdr
from fm_audio import FM_Audio

def callback(samples, sdr):
    global r
    r += 1
    if r > 4:
        sdr.cancel_read_async()
    else:
        fm_audio.fm_demod(samples, F_SAMPLE, N_SAMPLE)

def main(audio):

    sdr = RtlSdr()
    sdr.sample_rate = F_SAMPLE
    sdr.center_freq = STATION
    sdr.gain = 'auto'
    sdr.read_samples_async(callback, N_SAMPLE)
    sdr.close()

    audio.close()

if __name__ == "__main__":
    r = 0

    F_SAMPLE = 228e3
    N_SAMPLE = int(512*512)
    STATION = float(sys.argv[1]) * 1e6

    fm_audio = FM_Audio(F_SAMPLE, N_SAMPLE)

    main(fm_audio)
