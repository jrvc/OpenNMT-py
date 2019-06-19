# !/bin/bash 

export CUDA_VISIBLE_DEVICES=2,3

# Prepare data
#cd data
#source ./prep-data.sh

# Train
ONMT=`pwd`
SAVE_PATH=$ONMT/model/demo
mkdir -p $SAVE_PATH

#python train.py -data data/sample_data/m30k.de-cs \
#            data/sample_data/m30k.fr-cs \
#        -src_tgt de-cs fr-cs \
#        -save_model ${SAVE_PATH}/MULTILINGUAL_MULTI \
#        -use_attention_bridge \
#        -attention_heads 10 \
#        -rnn_size 512 \
#        -decoder_type rnn \
#        -enc_layers 2 \
#        -dec_layers 2 \
#        -word_vec_size 512 \
#        -global_attention mlp \
#        -train_steps 10000 \
#        -valid_steps 2000 \
#        -optim adam \
#        -learning_rate 0.0002 \
#        -batch_size 256 \
#        -save_checkpoint_steps 10000 \
#        -world_size 2 -gpu_ranks 0 1

# Translate          -tgt data/sample_data/test_2016_flickr.lc.norm.tok.cs \          -verbose
python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL2_step_10000.pt \
         -src_lang fr \
         -src data/sample_data/test_2016_flickr.lc.norm.tok.bpe.fr \
         -tgt_lang cs \
         -use_attention_bridge \
         -output ${SAVE_PATH}/MULTILINGUAL_prediction_fr-cs.bpe.txt \
         -gpu 0

testref=data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.cs
cat ${SAVE_PATH}/MULTILINGUAL_prediction_fr-cs.bpe.txt | sed -E 's/(@@ )|(@@ ?$)//g' | sacrebleu $testref

