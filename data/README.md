



> python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000

> python train.py -data data/data -save_model /n/rush_lab/data/tmp_ -gpuid 0 -rnn_size 100 -word_vec_size 50 -layers 1 -train_steps 100 -optim adam  -learning_rate 0.001

```
# multi-task
python train.py -data data/data data/data -src_tgt en-de fr-de -save_model /tmp/test_multitask -rnn_size 100 -word_vec_size 50 -layers 1 -train_steps 100 -optim adam  -learning_rate 0.001
```

#### Getting a toy multi source/target dataset
```
mkdir multi30k && cd multi30k
git clone git@github.com:multi30k/dataset.git

# path to your opennmt-py
ONMT=~/projects/mtm/OpenNMT-py

# run preprocessing on all pairs
# Note here we assume that the extracted vocab will always be the same
# every time we process the language
DATADIR=~/projects/mtm/multi30k/dataset/data/task1/tok
OUTPUT_DIR=~/projects/mtm/sample_data
mkdir $OUTPUT_DIR && cd $OUTPUT_DIR
for src_lang in de en fr cs
do
  for trg_lang in de en fr cs
  do
    SAVEDIR=$OUTPUT_DIR/${src_lang}-${trg_lang}
    mkdir $SAVEDIR
    src_train_file=$DATADIR/train.lc.norm.tok.${src_lang}
    trg_train_file=$DATADIR/train.lc.norm.tok.${trg_lang}
    src_valid_file=$DATADIR/val.lc.norm.tok.${src_lang}
    trg_valid_file=$DATADIR/val.lc.norm.tok.${trg_lang}
    python $ONMT/preprocess.py \
      -train_src $src_train_file \
      -train_tgt $trg_train_file \
      -valid_src $src_valid_file \
      -valid_tgt $trg_valid_file \
      -save_data $SAVEDIR/data \
      -src_vocab_size 10000 \
      -tgt_vocab_size 10000
  done
done

```




