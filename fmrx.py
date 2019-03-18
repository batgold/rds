#!/usr/bin/python
import sys
import pickle
import demod
import player
import rtlsdr
import argparse
import numpy as np
import constants as co
import multiprocessing #geeksforgeeks.com/multiprocessing

audio_que = multiprocessing.Queue()
demod_que = multiprocessing.Queue()
max_seconds = 2

def read_sdr(station):
    sdr = rtlsdr.RtlSdr()
    sdr.gain = 'auto'
    sdr.sample_rate = co.fs
    sdr.center_freq = float(station) * 1e6
    sdr.read_samples_async(sdr_callback, co.ns)
    sdr.close()

@rtlsdr.limit_time(max_seconds)
def sdr_callback(samples, sdr):
    process_samples(samples)
    write_file(samples)

def process_samples(samples):
    fm = demod_fm(samples)
    audio_que.put(fm)
    demod_que.put(fm)

def demod_fm(x):
    x = x[1:] * np.conj(x[:-1])
    fm = np.angle(x)
    return fm

def write_file(samples):
    """Open & write to file multiple times"""
    with open('rds_file', 'rb') as f:
        data = pickle.load(f)

    data.append(samples)

    with open('rds_file', 'wb') as f:
        pickle.dump(data, f)

def read_file(filename):
    """Open file and parse multiple sample sets"""
    with open(filename, 'rb') as f:
        samples = pickle.load(f)

    data_list = [list(n) for n in samples]

    for data in data_list:
        process_samples(data)

def clear_file():
    with open('rds_file', 'wb') as f:
        pickle.dump([], f)

def read_input():
    parser = argparse.ArgumentParser()
    file_arg = parser.add_mutually_exclusive_group()
    rtlsdr_arg = parser.add_mutually_exclusive_group()
    file_arg.add_argument(
        '-f', '--file', nargs=1, help='read file')
    rtlsdr_arg.add_argument(
        '-r', '--rtlsdr', nargs=1, help='live stream from rtlsdr')

    args = parser.parse_args()
    return args

def main():
    inputs = read_input()

    audio_proc = multiprocessing.Process(target=player.receive, args=(audio_que,))
    demod_proc = multiprocessing.Process(target=demod.receive, args=(demod_que,))

    audio_proc.start()
    demod_proc.start()

    if inputs.file:
        read_file(inputs.file[0])

    elif inputs.rtlsdr:
        clear_file()
        read_sdr(inputs.rtlsdr[0])

    audio_que.put(None)
    demod_que.put(None)

    audio_proc.join()
    demod_proc.join()

if __name__ == "__main__":
    main()
