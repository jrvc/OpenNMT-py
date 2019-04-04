#!/bin/bash

#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 03:00:00
#SBATCH --mem=30000
#SBATCH -J gpu_job
#SBATCH -o gpu_job.out.%j
#SBATCH -e gpu_job.err.%j
#SBATCH --gres=gpu:k80:1
#SBATCH

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9

DATADIR=/homeappl/home/celikkan/DATA/WMT18_en-fi
SRCDIR=/homeappl/home/celikkan/Github/OpenNMT-py

python $SRCDIR/preprocess.py -train_src $DATADIR/original/train/train.en -train_tgt $DATADIR/original/train/train.fi -valid_src $DATADIR/original/dev/dev.en -valid_tgt $DATADIR/original/dev/dev.fi -save_data $DATADIR/opennmt/en-fi

