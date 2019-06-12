#!/bin/bash


# get the multi30k dataset
if [ ! -d multi30k ]; then
    mkdir multi30k && cd multi30k
    git clone git@github.com:multi30k/dataset.git
    cd ..
fi

ONMT=`pwd`/..

DATADIR=`pwd`/multi30k/dataset/data/task1/tok
OUTPUT_DIR=`pwd`/sample_data

mkdir -p $OUTPUT_DIR && cd $OUTPUT_DIR


# BPE encodings
pip install subword-nmt
#RECALL: The original segmentation can be restored with the replacement:
#            sed -r 's/(@@ )|(@@ ?$)//g'
train_file=$DATADIR/train.lc.norm.tok
bpe_trainfile=$OUTPUT_DIR/train.lc.norm.tok.bpe10k

valid_file=$DATADIR/val.lc.norm.tok
bpe_validfile=$OUTPUT_DIR/val.lc.norm.tok.bpe10k

test_file=$DATADIR/test_2016_flickr.lc.norm.tok
bpe_testfile=$OUTPUT_DIR/test_2016_flickr.lc.norm.tok.bpe10k

num_operations='80000' # approx 10k per language  
codes_file=$OUTPUT_DIR/codecs40k 
vocab_file=$OUTPUT_DIR/bpe_vocab


# Learn BPE on all the training text, and get resulting vocabulary for each:
subword-nmt learn-joint-bpe-and-vocab \
         --input ${train_file}.{cs,de,en,fr} \
         -s ${num_operations} \
         -o ${codes_file} \
         --write-vocabulary ${vocab_file}.{cs,de,en,fr}

for lang in cs de en fr
do
  # re-apply byte pair encoding with vocabulary filter. # for test/dev data, re-use the same options.
  subword-nmt apply-bpe -c ${codes_file} --vocabulary ${vocab_file}.${lang} --vocabulary-threshold 20 < ${train_file}.${lang} > ${bpe_trainfile}.${lang}
  subword-nmt apply-bpe -c ${codes_file} --vocabulary ${vocab_file}.${lang} --vocabulary-threshold 20  < ${test_file}.${lang}  > ${bpe_testfile}.${lang}
  subword-nmt apply-bpe -c ${codes_file} --vocabulary ${vocab_file}.${lang} --vocabulary-threshold 20 < ${valid_file}.${lang} > ${bpe_validfile}.${lang}

done

# Preprocess with ONMT
# here we use the vocabularies constructed with subword-nmt
ALL_SAVE_DATA=""
for src_lang in de en fr cs
do
  for tgt_lang in de en fr cs
  do
    # no need for repeated data
    if [ ! -f $OUTPUT_DIR/m30k.${tgt_lang}-${src_lang}_2.vocab.pt ]; then
      # preprocess
      echo -e "\n Preprocessing language pair $src_lang - $tgt_lang "
      SAVEDATA=$OUTPUT_DIR/m30k.${src_lang}-${tgt_lang}
        #ALL_SAVE_DATA="$SAVEDATA $ALL_SAVE_DATA"
      src_train_file=${bpe_trainfile}.${src_lang}
      tgt_train_file=${bpe_trainfile}.${tgt_lang}
      src_valid_file=${bpe_validfile}.${src_lang}
      tgt_valid_file=${bpe_validfile}.${tgt_lang}
      python $ONMT/preprocess.py \
        -train_src $src_train_file \
        -train_tgt $tgt_train_file \
        -valid_src $src_valid_file \
        -valid_tgt $tgt_valid_file \
        -save_data ${SAVEDATA} \
        -src_vocab ${vocab_file}.${src_lang} \
        -tgt_vocab ${vocab_file}.${tgt_lang}
      fi
  done
done

# ---------------------------------------------------------------------------------------------
#     CREATE A VOCAB. FOR EACH LANGUAGE, USING ALL OF THE DATASETS FOR THAT LANGUAGE.
#----------------------------------------------------------------------------------------------
python $ONMT/preprocess_build_vocab.py \
    -train_dataset_prefixes $OUTPUT_DIR 

# Fix vocabs
python $ONMT/fix_vocab.py -l de en fr cs -b $OUTPUT_DIR -v m30k -n $OUTPUT_DIR

cd $ONMT
