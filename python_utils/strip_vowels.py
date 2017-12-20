from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys

ROOT_PATH=os.path.expanduser('~')
TACOTRON_PATH=os.path.join(ROOT_PATH, "tacotron")
STRESS=os.path.join(ROOT_PATH, "get-stress")

def strip_vowels(filename):
    with open(filename) as f:
        trs = [l.strip().split('|') for l in f]

    no_vowel_trs = [[' '.join([w for w in t.split(' ') if (w[:5] != 'VOWEL' and w != 'SIL')])
                     for t in tr]
                    for tr in trs]
    for tr in no_vowel_trs:
        print('|'.join(tr))

if __name__ == "__main__":
    strip_vowels(sys.argv[1])
