#!/bin/bash

path=~/tacotron/LJSpeech-1.0/wavs

for i in $path/LJ*.wav; do
  file=${i##*/}
  printf "%s-001 LJ-1\n" ${file%.wav}
done
