#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == "0" ]; then
    echo "TRAINING BASELINE"
    CHECKPOINT_NAME=0511-base
    DATA_PREFIX=9R-baseline

elif [ $1 == "1" ]; then
    echo "TRAINING MOTIFNET"
    CHECKPOINT_NAME=0511-reef0.99
    DATA_PREFIX=9R-reef0.99
fi

VG_SGG_FN=${DATA_PREFIX}-VG-SG.h5
VG_SGG_DICT_FN=${DATA_PREFIX}-VG-SG-dicts.json
#HACK: replace lines in the config file with dataset names
sed -i "s/VG_SGG_FN.*/VG_SGG_FN = stanford_path('$VG_SGG_FN')/" config.py
sed -i "s/VG_SGG_DICT_FN.*/VG_SGG_DICT_FN = stanford_path('$VG_SGG_DICT_FN')/" config.py
python models/train_rels.py \
    -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -save_dir checkpoints/$CHECKPOINT_NAME \
    -nepoch 50 -use_bias

