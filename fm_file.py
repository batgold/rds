#!/usr/bin/env python
import sys
import pickle
import rds
import audio
from constants import fs, ns

def main():
    with open(filename, 'rb') as f:
        samples = pickle.load(f)

    rds.demod(samples, station=104.5e6)
    #fm_audio = Audio(Fs, Ns)
    #fm_audio.demod(samples)
    #fm_audio.close()

if __name__ == "__main__":
    filename = sys.argv[1]
    main()
