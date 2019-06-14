# !/bin/bash 

# Prepare data
cd data
source ./prep-data.sh

# Train
ONMT=`pwd`
SAVE_PATH=$ONMT/model/demo
mkdir -p $SAVE_PATH
python train.py -data data/sample_data/m30k.de-cs \
            data/sample_data/m30k.fr-cs \
        -src_tgt de-cs fr-cs \
        -save_model ${SAVE_PATH}/MULTILINGUAL \
        -use_attention_bridge \
        -attention_heads 10 \
        -rnn_size 16 \
        -rnn_type GRU \
        -decoder_type rnn \
        -enc_layers 2 \
        -dec_layers 2 \
        -word_vec_size 16 \
        -global_attention mlp \
        -train_steps 10 \
        -valid_steps 10 \
        -optim adam \
        -learning_rate 0.0002 \
        -batch_size 1 \
        -save_checkpoint_steps 10000

# Translate
python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_step_10.pt \
         -src_lang fr \
         -src data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.fr \
         -tgt_lang cs \
         -tgt data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.cs \
         -use_attention_bridge \
         -output ${SAVE_PATH}/MULTILINGUAL_prediction_fr-cs.txt \
         -verbose

