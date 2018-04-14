#!/usr/bin/python

import numpy as nmp

def bit2int(bits):
    word = 0
    for bit in bits:
        word = (word<<1) | bit
    return int(word)

def prog_id(A):
    """Program Identification"""
    K = 4096
    W = 21672
    pi_int = bit2int(A)
    if pi_int >= W:
        pi0 = 'W'
        pi1 = W
    else:
        pi0 = 'K'
        pi1 = K
    pi2 = nmp.floor_divide(pi_int - pi1, 676)
    pi3 = nmp.floor_divide(pi_int - pi1 - pi2*676, 26)
    pi4 = pi_int - pi1 - pi2*676 - pi3*26
    return pi0 + chr(pi2+65) + chr(pi3+65) + chr(pi4+65)

def syndrome(bits):
    syn = 0
    for n, bit in enumerate(bits):
        if bit:
            print syn, H[n]
            syn = nmp.bitwise_xor(syn, H[n])
            print syn
    return syn

def rx():
    r = []
    ad = 0
    for x in HT:
        tmp = C ^ x
        #syn = nmp.mod(tmp,2)
        syn = bin(tmp).count('1')
        syn = nmp.mod(syn,2)
        r.append(syn)
    return r


H = [512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795]
HT = [33595239, 16797619, 8435902, 4244280, 2122140, 1093545, 575667, 326462, 163231, 81615]

#ben KQED
c = [False, False, True, True, True, False, True, False, True, False, True,
        False, True, False, True, True, False, True, True, True, False, True,
        True, False, False, False]
#from french guy
#c = [False, True, False, False, True, False, True, False, False, True, False, False, True, True, False, True, False, True, False, False, True, False, True, False, True, False]

#ben KQED offset
c = [False, True, True, True, False, True, False, True, False, True,
        False, True, False, True, True, False, True, True, True, False, True,
        True, False, False, False, False]
C = bit2int(c)

S = syndrome(c)
SA = 984
print "S=", S
print bin(S)

#r = rx()
#R = bit2int(r)
#print "R=", R
#print bin(R)

print "ERR=",S^SA
print bin(S^SA)
#Q = R ^ S
#print "Q=", Q
#print bin(Q)

msg = prog_id(c[0:16])
print msg


