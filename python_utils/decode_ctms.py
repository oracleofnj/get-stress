import os
import sys
import json
from cleaners import *

ROOT_PATH="/home/jss2272"
TACOTRON_PATH=os.path.join(ROOT_PATH, "tacotron")
STRESS=os.path.join(ROOT_PATH, "get-stress")

def read_phones(filename):
  phone_ids = {}
  with open(filename) as f:
    for line in f:
      phone, phoneid = line.strip().split(' ')
      phone_ids[int(phoneid)] = phone
  return phone_ids

def read_ctm(filename, phone_ids):
  utts = {}
  with open(filename) as f:
    for line in f:
      utt_id, _, start_time, end_time, phone_id = line.strip().split(' ')
      if utt_id not in utts:
        utts[utt_id] = []
      utts[utt_id].append({'start_time': float(start_time), 'duration': float(end_time), 'phone': phone_ids[int(phone_id)]})
  return utts


if __name__ == "__main__":
  phone_ids = read_phones(os.path.join(STRESS, sys.argv[1]))
  utts = read_ctm(os.path.join(STRESS, sys.argv[2]), phone_ids)
  print(json.dumps(utts, indent=2))




