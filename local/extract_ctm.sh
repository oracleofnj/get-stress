#!/bin/bash

for i in exp/mono_ali/ali.*.gz; do
  $KALDI_ROOT/src/bin/ali-to-phones --ctm-output exp/mono/final.mdl ark:"gunzip -c $i|" - > ${i%.gz}.ctm
done
