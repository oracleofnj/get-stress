#!/bin/bash

mkdir -p data/train
python python_utils/make_text.py LJSpeech-1.0/metadata.csv > data/train/text

