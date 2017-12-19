#!/bin/bash

path=/home/jss2272/tacotron/LJSpeech-1.0/wavs

for i in $path/LJ*.wav; do
  file=${i##*/}
  printf "%s %s\n" ${file%.wav} $i
done
