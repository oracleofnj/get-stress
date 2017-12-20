from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys
from clustering_utils_new import make_vowel_clusters

ROOT_PATH=os.path.expanduser('~')
TACOTRON_PATH=os.path.join(ROOT_PATH, "tacotron")
STRESS=os.path.join(ROOT_PATH, "get-stress")

if __name__ == "__main__":
    make_vowel_clusters(
        os.path.join(STRESS, sys.argv[1]),     # all_alignments.json
        os.path.join(STRESS, sys.argv[2]),     # numpy_features.npz
        os.path.join(STRESS, sys.argv[3]),     # vowel_models.pkl
    )
