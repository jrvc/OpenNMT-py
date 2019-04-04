#!/bin/bash

#SBATCH -J onmt_prep_enfi
#SBATCH -o onmt_prep_enfi.out.%j
#SBATCH -e onmt_prep_enfi.err.%j
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 03:00:00
#SBATCH --mem=10000
#SBATCH --gres=gpu:k80:1
#SBATCH

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9

INDIR=/homeappl/home/celikkan/DATA/WMT18_en-fi/original
OUTDIR=/homeappl/home/celikkan/DATA/WMT18_en-fi/opennmt
SRCDIR=/homeappl/home/celikkan/Github/OpenNMT-py

python $SRCDIR/preprocess.py -train_src $INDIR/train/train.en -train_tgt $INDIR/train/train.fi -valid_src $INDIR/dev/dev.en -valid_tgt $INDIR/dev/dev.fi -save_data $OUTDIR/en-fi -max_shard_size 100000000

