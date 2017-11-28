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
pyenv global general

echoerr() { echo "$@" 1>&2; }
set -e -o pipefail -u

nj=8
decode_nj=8    # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=10

. utils/parse_options.sh # accept options

if [ $stage -le 1 ]; then
  echo "6998: Beginning stage 1"
  echoerr "6998: Beginning stage 1"
  mkdir -p data/train
  . local/prepare_segments.sh > data/train/segments_all
  . local/prepare_wav_scp.sh > data/train/wav.scp
  . local/prepare_utt2spk.sh > data/train/utt2spk_all
  . local/prepare_textdata.sh
  . utils/fix_data_dir.sh data/train
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
  #  more than our normal 30 jobs.]
#  for dset in dev test train; do
#    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
#  done
fi
echo "6998: Completed stage 1"
echoerr "6998: Completed stage 1"

. path.sh
pyenv global system

if [ $stage -le 2 ]; then
  echo "6998: Beginning stage 2"
  echoerr "6998: Beginning stage 2"
  local/prepare_dict.sh
fi
echo "6998: Completed stage 2"
echoerr "6998: Completed stage 2"

if [ $stage -le 3 ]; then
  echo "6998: Beginning stage 3"
  echoerr "6998: Beginning stage 3"
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
fi
echo "6998: Completed stage 3"
echoerr "6998: Completed stage 3"

if [ $stage -le 4 ]; then
  echo "6998: Beginning stage 4"
  echoerr "6998: Beginning stage 4"
  local/lj_train_lm.sh
fi
echo "6998: Completed stage 4"
echoerr "6998: Completed stage 4"

if [ $stage -le 5 ]; then
  echo "6998: Beginning stage 5"
  echoerr "6998: Beginning stage 5"
  local/format_lms.sh
fi
echo "6998: Completed stage 5"
echoerr "6998: Completed stage 5"

# Feature extraction
if [ $stage -le 6 ]; then
  echo "6998: Beginning stage 6"
  echoerr "6998: Beginning stage 6"
  dir=data/train
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $dir
  steps/compute_cmvn_stats.sh $dir
fi
echo "6998: Completed stage 6"
echoerr "6998: Completed stage 6"

# Now we have 212 hours of training data.
# Well create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 7 ]; then
  echo "6998: Beginning stage 7"
  echoerr "6998: Beginning stage 7"
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  utils/data/remove_dup_utts.sh 10 data/train_10kshort data/train_10kshort_nodup
fi
echo "6998: Completed stage 7"
echoerr "6998: Completed stage 7"

# Train
if [ $stage -le 8 ]; then
  echo "6998: Beginning stage 8"
  echoerr "6998: Beginning stage 8"
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/train_10kshort_nodup data/lang_nosp exp/mono
fi
echo "6998: Completed stage 8"
echoerr "6998: Completed stage 8"

if [ $stage -le 9 ]; then
  echo "6998: Beginning stage 9"
  echoerr "6998: Beginning stage 9"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/mono exp/mono_ali
  #steps/train_deltas.sh --cmd "$train_cmd" \
  #  2500 30000 data/train data/lang_nosp exp/mono_ali exp/tri1
fi
echo "6998: Completed stage 9"
echoerr "6998: Completed stage 9"

if [ $stage -le 10 ]; then
  echo "6998: Beginning stage 10"
  echoerr "6998: Beginning stage 10"
  . local/extract_ctm.sh
fi
echo "6998: Completed stage 10"
echoerr "6998: Completed stage 10"

exit

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
