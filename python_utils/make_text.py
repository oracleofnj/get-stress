import os
import sys
from cleaners import *

ROOT_PATH="/home/jss2272"
TACOTRON_PATH=os.path.join(ROOT_PATH, "tacotron")
STRESS=os.path.join(ROOT_PATH, "get-stress")

def clean_line(text):
  text_charsonly = re.sub(r"[^a-zA-Z']", " ", text)
  return english_cleaners(text_charsonly)

def convert_line(line):
  utt_id, unnorm_text, norm_text = line.strip().split('|')
  return utt_id, clean_line(norm_text)

def convert_metadata(filename):
  with open(filename) as f:
    for line in f:
      utt_id, cleaned_text = convert_line(line)
      print('{0}-001 {1}'.format(utt_id, cleaned_text))

if __name__ == "__main__":
  convert_metadata(os.path.join(TACOTRON_PATH, sys.argv[1]))



