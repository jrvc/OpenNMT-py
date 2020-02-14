# !/bin/bash 

# Train
ONMT=`pwd`
SAVE_PATH=$ONMT/model/demo
mkdir -p $SAVE_PATH

python train.py -data data/sample_data/m30k.de-cs \
            data/sample_data/m30k.fr-cs \
        -src_tgt de-cs fr-cs \
        -save_model ${SAVE_PATH}/MULTILINGUAL_MULTI \
        -use_attention_bridge \
        -attention_heads 10 \
        -rnn_size 256 \
        -decoder_type rnn \
        -enc_layers 1 \
        -dec_layers 1 \
        -word_vec_size 256 \
        -train_steps 10000 \
        -valid_steps 5000 \
        -optim adam \
        -learning_rate 0.0002 \
        -batch_size 128 \
        -save_checkpoint_steps 10000 \
        -model_type text

#        -world_size 1 -gpu_ranks 0
