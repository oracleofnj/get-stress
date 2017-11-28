import numpy as np
import os
import sys

ROOT_PATH="/home/jss2272"
TACOTRON_PATH=os.path.join(ROOT_PATH, "tacotron")
STRESS=os.path.join(ROOT_PATH, "get-stress")

def kaldi_to_npz(kaldifile):
    mats = {}
    with open(kaldifile) as f:
        active_matrix = []
        for line in f:
            words = line.strip().split(' ')
            if words[-1] == '[':
                matrix_name = words[0]
            elif words[-1] == ']':
                active_matrix.append([float(word) for word in words[:-1]])
                mats[matrix_name] = np.array(active_matrix, dtype=np.float32)
                active_matrix = []
            else:
                active_matrix.append([float(word) for word in words])
    return mats

if __name__ == "__main__":
  mats = kaldi_to_npz(os.path.join(STRESS, sys.argv[1]))
  np.savez_compressed(os.path.join(STRESS, sys.argv[2]), **mats)



