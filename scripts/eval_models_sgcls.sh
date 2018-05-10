#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"

    BASELINE_CHECKPOINT=motifnet-sgcls-baserels
    CHECKPOINT_FN=vgrel-12.tar
    python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$BASELINE_CHECKPOINT/$CHECKPOINT_FN -nepoch 50 -use_bias
    python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$BASELINE_CHECKPOINT/$CHECKPOINT_FN -nepoch 50 -use_bias
elif [ $1 == "1" ]; then

    REEF_CHECKPOINT=motifnet-sgcls-reefrels
    CHECKPOINT_FN=vgrel-12.tar
    echo "EVALING MOTIFNET"
    python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$REEF_CHECKPOINT/$CHECKPOINT_FN -nepoch 50 -use_bias
    python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/$REEF_CHECKPOINT/$CHECKPOINT_FN -nepoch 50 -use_bias
fi



