#!/bin/bash
#
# Adapted from the Tedlium example in Kaldi.
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Vincent Nguyen
#            2016  Johns Hopkins University (Author: Daniel Povey)
#
# Apache 2.0
#

export train_cmd=run.pl
export decode_cmd=run.pl
export TEDLIUM=../kaldi-trunk/egs/tedlium/s5_r2
export VOWEL_CLUSTER_MODEL=tedlium_8_clusters_python2.pkl

echoerr() { echo "$@" 1>&2; }
set -e -o pipefail -u

nj=1
decode_nj=1    # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.

. utils/parse_options.sh # accept options
. path.sh

if [ $# -lt 2 ] || [ $# -gt 2 ]; then
   echo "Usage: $0 <wav-path> <transcription-path>";
   exit 1;
fi

# Resample to 16k

EXTENSION=${1##*.}
ALL_BUT_EXTENSION=${1%.*}
INPUT_FILENAME=${ALL_BUT_EXTENSION##*/}
ANNOTATION_DIR=/tmp/$INPUT_FILENAME
MONO_WAV=${INPUT_FILENAME}_mono_16k.wav
mkdir -p $ANNOTATION_DIR

if [ "$EXTENSION" == "wav" ]; then
  sox $1 -c 1 -r 16000 $ANNOTATION_DIR/$MONO_WAV
else
  ffmpeg -i $1 $ANNOTATION_DIR/${INPUT_FILENAME}_orig.wav
  sox $ANNOTATION_DIR/${INPUT_FILENAME}_orig.wav -c 1 -r 16000 $ANNOTATION_DIR/$MONO_WAV
fi

# Make utt2spk
echo "$INPUT_FILENAME $INPUT_FILENAME" > $ANNOTATION_DIR/utt2spk

# Make spk2utt
utils/utt2spk_to_spk2utt.pl $ANNOTATION_DIR/utt2spk > $ANNOTATION_DIR/spk2utt

# Make segments
length="$(sox $ANNOTATION_DIR/$MONO_WAV -n stat 2>&1 | sed -n 's#^Length (seconds):[^0-9]*\([0-9.]*\)$#\1#p')"
echo "$INPUT_FILENAME $INPUT_FILENAME 0 $length" > $ANNOTATION_DIR/segments

# Make wav.scp
echo "$INPUT_FILENAME $ANNOTATION_DIR/$MONO_WAV" > $ANNOTATION_DIR/wav.scp

# Compute MFCCs
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $ANNOTATION_DIR
steps/compute_cmvn_stats.sh $ANNOTATION_DIR

TRANSCRIPTION="$(cat $2)"
echo "$INPUT_FILENAME $TRANSCRIPTION" > $ANNOTATION_DIR/text

# Create alignment
mkdir -p $ANNOTATION_DIR/ali
steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  $ANNOTATION_DIR $TEDLIUM/data/lang $TEDLIUM/exp/tri3 $ANNOTATION_DIR/ali

# Convert to phones
$KALDI_ROOT/src/bin/ali-to-phones \
  --ctm-output $ANNOTATION_DIR/ali/final.mdl \
  ark:"gunzip -c $ANNOTATION_DIR/ali/ali.1.gz|" - > $ANNOTATION_DIR/ali/ali.1.ctm

# Convert to JSON
python python_utils/decode_ctms.py \
  $TEDLIUM/data/lang/phones.txt \
  $ANNOTATION_DIR/ali/ali.1.ctm > $ANNOTATION_DIR/ali.json

# Make pitch features
mkdir -p $ANNOTATION_DIR/pitch
steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" \
  $ANNOTATION_DIR $ANNOTATION_DIR/log $ANNOTATION_DIR/pitch

# Convert to numpy
copy-matrix scp:$ANNOTATION_DIR/feats.scp ark,t:- > $ANNOTATION_DIR/pitch/textoutput.txt
python python_utils/kaldi_to_npz.py $ANNOTATION_DIR/pitch/textoutput.txt $ANNOTATION_DIR/pitch/numpy_features

# Add cluster labels
python python_utils/generate_one_transcription.py \
  $ANNOTATION_DIR/ali.json \
  $ANNOTATION_DIR/pitch/numpy_features.npz \
  $VOWEL_CLUSTER_MODEL \
  $ANNOTATION_DIR/clustered.csv

python python_utils/generate_plot.py \
  $ANNOTATION_DIR/ali.json \
  $ANNOTATION_DIR/pitch/numpy_features.npz \
  $ANNOTATION_DIR

# Print the cluster file
echo ""
echo
cat $ANNOTATION_DIR/clustered.csv

echo "Plot saved in $ANNOTATION_DIR/$INPUT_FILENAME.png"
exit
