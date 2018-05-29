#!/usr/bin/python
import math

fs = 228e3
fc = 57e3
fa = 44.1e3
fsym = 1187.5
ns = 512*512
rds_dec = 12
aud_dec = int(fs/fa)
tau = 75e-6         #de-emphasis time constant
pi = math.pi
parity = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
syndromes = {'A':984, 'B':980, 'C':604, 'D':600}
