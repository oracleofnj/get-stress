from __future__ import print_function
from __future__ import division
import os
import sys
from cleaners import *

ROOT_PATH=os.path.expanduser('~')
TACOTRON_PATH=os.path.join(ROOT_PATH, "tacotron")
STRESS=os.path.join(ROOT_PATH, "get-stress")

def convert_line(line):
  utt_id, unnorm_text, norm_text = line.strip().split('|')
  return utt_id

def get_utterances(filename):
  utterances = {}
  with open(filename) as f:
    for line in f:
      utt_id = convert_line(line)
      utterances['{0}-001'.format(utt_id)] = filename
  return utterances

def filter_file(filename, utterances):
  with open(filename) as f:
    for line in f:
      words = line.strip().split(' ')
      if words[0] in utterances:
        print(line.strip())

if __name__ == "__main__":
  utterances = get_utterances(os.path.join(TACOTRON_PATH, sys.argv[1]))
  filter_file(os.path.join(STRESS, sys.argv[2]), utterances)
