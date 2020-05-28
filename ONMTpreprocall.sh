#!/bin/bash


source env/bin/activate 
option=$1

h5PATH=/path/to/directory/where/preprocessed/textfiles/are/
outPATH=/path/to/directory/with/1wavfile-per-line/files/
 

#####     ENaudio -> DEtext     #####

if [ "$option" == 'all.ende' ]
then
mkdir -p ${outPATH}/all/ENaudio_DEtext

echo -e "\n"
python preprocess.py -save_config ${outPATH}/all/ENaudio_DEtext/preprocess.config \
                     -data_type audio \
                     -src_dir   ${outPATH} \
                     -train_src ${outPATH}/all.train.wavnames \
                     -train_tgt ${h5PATH}/all.train.de \
                     -valid_src ${outPATH}/all.dev.wavnames   \
                     -valid_tgt ${h5PATH}/all.dev.de   \
                     -tgt_vocab ${h5PATH}/all.vocab.de  \
                     -src_seq_length 150 \
                     -tgt_seq_length 100 \
                     -shard_size 5000 \
                     -n_mels 80 \
                     -save_data ${outPATH}/all/ENaudio_DEtext/data


echo -e "\n \n Preprocessing ENaudio -> DEtext for experim: all \n \n"
python preprocess.py -config ${outPATH}/all/ENaudio_DEtext/preprocess.config
fi


#####     ENaudio -> ENtext     #####

if [ "$option" == 'all.enen' ]
then

mkdir -p ${outPATH}/all/ENaudio_ENtext

echo -e "\n"
python preprocess.py -save_config ${outPATH}/all/ENaudio_ENtext/preprocess.config \
                     -data_type audio \
                     -src_dir   ${outPATH} \
                     -train_src ${outPATH}/all.train.wavnames \
                     -train_tgt ${h5PATH}/all.train.en \
                     -valid_src ${outPATH}/all.dev.wavnames   \
                     -valid_tgt ${h5PATH}/all.dev.en   \
                     -tgt_vocab ${h5PATH}/all.vocab.en  \
                     -src_seq_length 150 \
                     -tgt_seq_length 100 \
                     -shard_size 5000 \
                     -n_mels 80 \
                     -save_data ${outPATH}/all/ENaudio_ENtext/data

echo -e "\n \n Preprocessing ENaudio -> ENtext for experim: all \n \n"
python preprocess.py -config ${outPATH}/all/ENaudio_ENtext/preprocess.config

fi

if [ "$option" == 'text' ]
then
#####     ENtext -> DEtext     #####
mkdir -p ${outPATH}/all/ENtext_DEtext
src=en
tgt=de
python preprocess.py -save_config ${outPATH}/all/ENtext_DEtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.train.${src} \
                     -train_tgt ${h5PATH}/all.train.${tgt} \
                     -valid_src ${h5PATH}/all.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data  ${outPATH}/all/ENtext_DEtext/data

python preprocess.py -config  ${outPATH}/all/ENtext_DEtext/preprocess.config




#####     DEtext -> ENtext     #####
mkdir -p ${outPATH}/all/DEtext_ENtext
src=de
tgt=en
python preprocess.py -save_config ${outPATH}/all/DEtext_ENtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.train.${src} \
                     -train_tgt ${h5PATH}/all.train.${tgt} \
                     -valid_src ${h5PATH}/all.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data ${outPATH}/all/DEtext_ENtext/data                    

python preprocess.py -config  ${outPATH}/all/DEtext_ENtext/preprocess.config

#####     ENtext -> ENtext     #####
mkdir -p ${outPATH}/all/ENtext_ENtext
src=en
tgt=en
python preprocess.py -save_config ${outPATH}/all/ENtext_ENtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.train.${src} \
                     -train_tgt ${h5PATH}/all.train.${tgt} \
                     -valid_src ${h5PATH}/all.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data ${outPATH}/all/ENtext_ENtext/data

python preprocess.py -config  ${outPATH}/all/ENtext_ENtext/preprocess.config

#####     DEtext -> DEtext     #####
mkdir -p ${outPATH}/all/DEtext_DEtext
src=de
tgt=en
python preprocess.py -save_config ${outPATH}/all/DEtext_DEtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.train.${src} \
                     -train_tgt ${h5PATH}/all.train.${tgt} \
                     -valid_src ${h5PATH}/all.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data ${outPATH}/all/DEtext_DEtext/data

python preprocess.py -config  ${outPATH}/all/DEtext_DEtext/preprocess.config


fi


if [ "$option" == 'text.char' ]
then
#####     ENtext -> DEtext     #####
mkdir -p ${outPATH}/all.char/ENtext_DEtext
src=en
tgt=de
python preprocess.py -save_config ${outPATH}/all.char/ENtext_DEtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.char.train.${src} \
                     -train_tgt ${h5PATH}/all.char.train.${tgt} \
                     -valid_src ${h5PATH}/all.char.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.char.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.char.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.char.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data  ${outPATH}/all.char/ENtext_DEtext/data

python preprocess.py -config  ${outPATH}/all.char/ENtext_DEtext/preprocess.config




#####     DEtext -> ENtext     #####
mkdir -p ${outPATH}/all.char/DEtext_ENtext
src=de
tgt=en
python preprocess.py -save_config ${outPATH}/all.char/DEtext_ENtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.char.train.${src} \
                     -train_tgt ${h5PATH}/all.char.train.${tgt} \
                     -valid_src ${h5PATH}/all.char.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.char.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.char.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.char.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data ${outPATH}/all.char/DEtext_ENtext/data                    

python preprocess.py -config  ${outPATH}/all.char/DEtext_ENtext/preprocess.config

#####     ENtext -> ENtext     #####
mkdir -p ${outPATH}/all.char/ENtext_ENtext
src=en
tgt=en
python preprocess.py -save_config ${outPATH}/all.char/ENtext_ENtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.char.train.${src} \
                     -train_tgt ${h5PATH}/all.char.train.${tgt} \
                     -valid_src ${h5PATH}/all.char.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.char.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.char.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.char.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data ${outPATH}/all.char/ENtext_ENtext/data

python preprocess.py -config  ${outPATH}/all.char/ENtext_ENtext/preprocess.config

#####     DEtext -> DEtext     #####
mkdir -p ${outPATH}/all.char/DEtext_DEtext
src=de
tgt=en
python preprocess.py -save_config ${outPATH}/all.char/DEtext_DEtext/preprocess.config \
                     -data_type text \
                     -train_src ${h5PATH}/all.char.train.${src} \
                     -train_tgt ${h5PATH}/all.char.train.${tgt} \
                     -valid_src ${h5PATH}/all.char.dev.${src}   \
                     -valid_tgt ${h5PATH}/all.char.dev.${tgt}   \
                     -src_vocab ${h5PATH}/all.char.vocab.${src}  \
                     -tgt_vocab ${h5PATH}/all.char.vocab.${tgt}  \
                     -src_seq_length 100 \
                     -tgt_seq_length 100 \
                     -shard_size 50000 \
                     -save_data ${outPATH}/all.char/DEtext_DEtext/data

python preprocess.py -config  ${outPATH}/all.char/DEtext_DEtext/preprocess.config


fi
