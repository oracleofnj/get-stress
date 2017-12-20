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

echoerr() { echo "$@" 1>&2; }
set -e -o pipefail -u

nj=1
decode_nj=1    # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.

. utils/parse_options.sh # accept options
. path.sh

if [ $# -lt 1 ] || [ $# -gt 1 ]; then
   echo "Usage: $0 <wav-path>";
   exit 1;
fi

# Resample to 16k

ANNOTATION_DIR=/tmp/${1%.wav}
MONO_WAV=${1%.wav}_mono_16k.wav
mkdir -p $ANNOTATION_DIR
sox $1 -c 1 -r 16000 $ANNOTATION_DIR/$MONO_WAV



exit

if [ $stage -le 14 ]; then
  echo "6998: Beginning stage 14"
  echoerr "6998: Beginning stage 14"
  mkdir -p exp/tri3_ali
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $TEDLIUM/data/train $TEDLIUM/data/lang $TEDLIUM/exp/tri3 exp/tri3_ali
fi
echo "6998: Completed stage 14"
echoerr "6998: Completed stage 14"

if [ $stage -le 15 ]; then
  echo "6998: Beginning stage 15"
  echoerr "6998: Beginning stage 15"
  for i in exp/tri3_ali/ali.*.gz; do
    $KALDI_ROOT/src/bin/ali-to-phones --ctm-output exp/tri3_ali/final.mdl ark:"gunzip -c $i|" - > ${i%.gz}.ctm
  done
fi
echo "6998: Completed stage 15"
echoerr "6998: Completed stage 15"

if [ $stage -le 16 ]; then
  echo "6998: Beginning stage 16"
  echoerr "6998: Beginning stage 16"
  mkdir -p tedlium_pitch
  steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" $TEDLIUM/data/train tedlium_pitch/log tedlium_pitch
  . utils/fix_data_dir.sh $TEDLIUM/data/train
fi
echo "6998: Completed stage 16"
echoerr "6998: Completed stage 16"

if [ $stage -le 17 ]; then
  echo "6998: Beginning stage 17"
  echoerr "6998: Beginning stage 17"
  dir=$TEDLIUM/data/train
  copy-matrix scp:$dir/feats.scp ark,t:- > tedlium_pitch/textoutput.txt
fi
echo "6998: Completed stage 17"
echoerr "6998: Completed stage 17"

if [ $stage -le 18 ]; then
  echo "6998: Beginning stage 18"
  echoerr "6998: Beginning stage 18"
  dir=data/train
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $dir
  steps/compute_cmvn_stats.sh $dir
fi
echo "6998: Completed stage 18"
echoerr "6998: Completed stage 18"

if [ $stage -le 19 ]; then
  echo "6998: Beginning stage 19"
  echoerr "6998: Beginning stage 19"
  mkdir -p exp/tri3_lj_ali
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train $TEDLIUM/data/lang $TEDLIUM/exp/tri3 exp/tri3_lj_ali
fi
echo "6998: Completed stage 19"
echoerr "6998: Completed stage 19"

if [ $stage -le 20 ]; then
  echo "6998: Beginning stage 20"
  echoerr "6998: Beginning stage 20"
  for i in exp/tri3_lj_ali/ali.*.gz; do
    $KALDI_ROOT/src/bin/ali-to-phones --ctm-output exp/tri3_lj_ali/final.mdl ark:"gunzip -c $i|" - > ${i%.gz}.ctm
  done
fi
echo "6998: Completed stage 20"
echoerr "6998: Completed stage 20"

exit
pyenv global general

if [ $stage -le 21 ]; then
  echo "6998: Beginning stage 21"
  echoerr "6998: Beginning stage 21"
  dir=data/train
  python python_utils/kaldi_to_npz.py tedlium_pitch/textoutput.txt tedlium_pitch/numpy_features
fi
echo "6998: Completed stage 21"
echoerr "6998: Completed stage 21"

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 10 ]; then
  utils/mkgraph.sh data/lang_nosp exp/tri1 exp/tri1/graph_nosp

  # The slowest part about this decoding is the scoring, which we can't really
  # control as the bottleneck is the NIST tools.
  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri1/graph_nosp data/${dset} exp/tri1/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
       data/${dset} exp/tri1/decode_nosp_${dset} exp/tri1/decode_nosp_${dset}_rescore
  done
fi

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 11 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang_nosp exp/tri1_ali exp/tri2
fi

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 12 ]; then
  utils/mkgraph.sh data/lang_nosp exp/tri2 exp/tri2/graph_nosp
  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri2/graph_nosp data/${dset} exp/tri2/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
       data/${dset} exp/tri2/decode_nosp_${dset} exp/tri2/decode_nosp_${dset}_rescore
  done
fi

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 13 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp exp/tri2
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri2/pron_counts_nowb.txt \
    exp/tri2/sil_counts_nowb.txt \
    exp/tri2/pron_bigram_counts_nowb.txt data/local/dict
fi

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 14 ]; then
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp -rT data/lang data/lang_rescore
  cp data/lang_nosp/G.fst data/lang/
  cp data/lang_nosp_rescore/G.carpa data/lang_rescore/

  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph

  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri2/graph data/${dset} exp/tri2/decode_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset} exp/tri2/decode_${dset} exp/tri2/decode_${dset}_rescore
  done
fi

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 15 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2_ali exp/tri3

  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

#  for dset in dev test; do
#    # steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
#    #    exp/tri3/graph data/${dset} exp/tri3/decode_${dset}
#    # steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
#    #    data/${dset} exp/tri3/decode_${dset} exp/tri3/decode_${dset}_rescore
#  done
fi

# the following shows you how to insert a phone language model in place of <unk>
# and decode with that.
# local/run_unk_model.sh

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 16 ]; then
  # this does some data-cleaning.  It actually degrades the GMM-level results
  # slightly, but the cleaned data should be useful when we add the neural net and chain
  # systems.  If not we'll remove this stage.
  local/run_cleanup_segmentation.sh
fi


# TODO: xiaohui-zhang will add lexicon cleanup at some point.

echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."
if [ $stage -le 17 ]; then
  # This will only work if you have GPUs on your system (and note that it requires
  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
  local/chain/run_tdnn.sh
fi

# The nnet3 TDNN recipe:
# local/nnet3/run_tdnn.sh


# We removed the GMM+MMI stage that used to exist in the release-1 scripts,
# since the neural net training is more of interest.

echo "$0: success."
exit 0
