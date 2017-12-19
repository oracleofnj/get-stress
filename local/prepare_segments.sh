#!/bin/bash

path=~/tacotron/LJSpeech-1.0/wavs

for i in $path/LJ*.wav; do
  length="$(sox $i -n stat 2>&1 | sed -n 's#^Length (seconds):[^0-9]*\([0-9.]*\)$#\1#p')"
  file=${i##*/}
  printf "%s-001 %s 0.000 %s\n" {${file%.wav},${file%.wav},$length}
done
