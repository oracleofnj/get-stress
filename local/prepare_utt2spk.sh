#!/bin/bash

path=~/tacotron/LJSpeech-1.0/wavs

for i in $path/LJ*.wav; do
  file=${i##*/}
  printf "%s-001 %s\n" ${file%.wav} ${file%.wav}
done
