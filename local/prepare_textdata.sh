#!/bin/bash

mkdir -p data/train
python python_utils/make_text.py LJSpeech-1.0/metadata.csv > data/train/text
python python_utils/remove_empties.py LJSpeech-1.0/metadata.csv data/train/segments_all > data/train/segments
python python_utils/remove_empties.py LJSpeech-1.0/metadata.csv data/train/utt2spk_all > data/train/utt2spk
rm data/train/segments_all
rm data/train/utt2spk_all


