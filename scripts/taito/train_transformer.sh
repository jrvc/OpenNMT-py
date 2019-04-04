#!/bin/bash

#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH -J onmt_train
#SBATCH -o onmt_train.out.%j
#SBATCH -e onmt_train.err.%j
#SBATCH --gres=gpu:k80:1
#SBATCH

module purge
#module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9
module load gcc cuda python-env/3.6.3-ml

DATADIR=/homeappl/home/celikkan/DATA/WMT18_en-fi/opennmt
SRCDIR=/homeappl/home/celikkan/Github/OpenNMT-py
MODELDIR=$SRCDIR/models

python  $SRCDIR/train.py -data $DATADIR/en-fi -save_model models \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000 -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 4 -gpu_ranks 0 1 2 3 

