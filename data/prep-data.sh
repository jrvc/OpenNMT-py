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

ALL_SAVE_DATA=""
for src_lang in de en fr cs
do
  for trg_lang in de en fr cs
  do
    SAVEDIR=$OUTPUT_DIR/${src_lang}-${trg_lang}
    mkdir -p $SAVEDIR
	SAVEDATA=$SAVEDIR/data
	ALL_SAVE_DATA="$SAVEDATA $ALL_SAVE_DATA"
    src_train_file=$DATADIR/train.lc.norm.tok.${src_lang}
    trg_train_file=$DATADIR/train.lc.norm.tok.${trg_lang}
    src_valid_file=$DATADIR/val.lc.norm.tok.${src_lang}
    trg_valid_file=$DATADIR/val.lc.norm.tok.${trg_lang}
    python $ONMT/preprocess.py \
      -train_src $src_train_file \
      -train_tgt $trg_train_file \
      -valid_src $src_valid_file \
      -valid_tgt $trg_valid_file \
      -save_data $SAVEDATA \
      -src_vocab_size 10000 \
      -tgt_vocab_size 10000
  done
done

python $ONMT/preprocess_build_vocab.py \
	-share_vocab \
	-train_dataset_prefixes $ALL_SAVE_DATA \
    -src_vocab_size 10000 \
    -tgt_vocab_size 10000
