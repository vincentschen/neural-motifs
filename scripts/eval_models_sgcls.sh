#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"
    CHECKPOINT_DIR=0511-base
    CHECKPOINT_FILE=vgrel-19.tar

elif [ $1 == "1" ]; then
    echo "EVALING MOTIFNET"
    CHECKPOINT_DIR=0511-reef0.99
    CHECKPOINT_FILE=vgrel-14.tar
fi

python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$CHECKPOINT_DIR/$CHECKPOINT_FILE -nepoch 50 -use_bias
python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$CHECKPOINT_DIR/$CHECKPOINT_FILE -nepoch 50 -use_bias


