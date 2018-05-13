#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == "0" ]; then
    echo "EVALING BASELINE"
    CHECKPOINT_DIR=0512-baseline
    CHECKPOINT_FILE=vgrel-5.tar
    DATA_PREFIX=10R-baseline

elif [ $1 == "1" ]; then
    echo "EVALING MOTIFNET"
    CHECKPOINT_DIR=0512-oracle
    CHECKPOINT_FILE=vgrel-15.tar
    DATA_PREFIX=10R-oracle

elif [ $1 == "2" ]; then
    echo "EVALING ORACLE"
    CHECKPOINT_DIR=0513-oracle
    CHECKPOINT_FILE=vgrel-11.tar
    DATA_PREFIX=10R-oracle
fi

VG_SGG_FN=${DATA_PREFIX}-VG-SG.h5
VG_SGG_DICT_FN=${DATA_PREFIX}-VG-SG-dicts.json
#HACK: replace lines in the config file with dataset names
sed -i "s/VG_SGG_FN.*/VG_SGG_FN = stanford_path('$VG_SGG_FN')/" config.py
sed -i "s/VG_SGG_DICT_FN.*/VG_SGG_DICT_FN = stanford_path('$VG_SGG_DICT_FN')/" config.py

python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$CHECKPOINT_DIR/$CHECKPOINT_FILE -nepoch 50 -use_bias
python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$CHECKPOINT_DIR/$CHECKPOINT_FILE -nepoch 50 -use_bias


