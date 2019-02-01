#!/bin/bash

if [ ! -d multi30k ]; then
    mkdir multi30k && cd multi30k
    git clone git@github.com:multi30k/dataset.git
    cd ..
fi

ONMT=`pwd`/..

DATADIR=`pwd`/multi30k/dataset/data/task1/tok
OUTPUT_DIR=`pwd`/sample_data

mkdir -p $OUTPUT_DIR && cd $OUTPUT_DIR

# create all language pairs
ALL_SAVE_DATA=""
for src_lang in de en fr cs
do
  for tgt_lang in de en fr cs
  do
    # no need for repeated data
    if [ ! -f $OUTPUT_DIR/m30k.${tgt_lang}-${src_lang}.vocab.pt ]; then
      # preprocess
        SAVEDATA=$OUTPUT_DIR/m30k.${src_lang}-${tgt_lang}
        #ALL_SAVE_DATA="$SAVEDATA $ALL_SAVE_DATA"
      src_train_file=$DATADIR/train.lc.norm.tok.${src_lang}
      tgt_train_file=$DATADIR/train.lc.norm.tok.${tgt_lang}
      src_valid_file=$DATADIR/val.lc.norm.tok.${src_lang}
      tgt_valid_file=$DATADIR/val.lc.norm.tok.${tgt_lang}
      python $ONMT/preprocess.py \
        -train_src $src_train_file \
        -train_tgt $tgt_train_file \
        -valid_src $src_valid_file \
        -valid_tgt $tgt_valid_file \
        -save_data $SAVEDATA \
        -src_vocab_size 10000 \
        -tgt_vocab_size 10000
      fi
  done
done

# ---------------------------------------------------------------------------------------------
#     CREATE A VOCAB. FOR EACH LANGUAGE, USING ALL OF THAT LANGUAGE DATASETS.
#----------------------------------------------------------------------------------------------
python $ONMT/preprocess_build_vocab.py \
    -train_dataset_prefixes $OUTPUT_DIR \
    -share_vocab \
    -src_vocab_size 10000 \
    -tgt_vocab_size 10000


#
