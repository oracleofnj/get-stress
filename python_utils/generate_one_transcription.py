from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys
from clustering_utils_new import make_transcriptions_subsample

if __name__ == "__main__":
    make_transcriptions_subsample(
        sys.argv[1],     # all_alignments.json
        sys.argv[2],     # numpy_features.npz
        sys.argv[3],     # vowel_models.pkl
        sys.argv[4],     # sample_transcriptions.csv
        1.0,
        strip_four=False
    )
