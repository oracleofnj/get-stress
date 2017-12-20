from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import json
import pickle
import os
import sys
from clustering_utils_new import plot_utterances


if __name__ == "__main__":
    plot_utterances(
        sys.argv[1],     # all_alignments.json
        sys.argv[2],     # numpy_features.npz
        sys.argv[3],     # /tmp/foo
    )
