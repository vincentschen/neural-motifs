#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

CHECKPOINT_NAME=0512-baseline
DATA_PREFIX=10R-baseline-50sampled
echo "TRAINING MOTIFNET ${CHECKPOINT_NAME}"
VG_SGG_FN=${DATA_PREFIX}-VG-SG.h5
VG_SGG_DICT_FN=${DATA_PREFIX}-VG-SG-dicts.json
#HACK: replace lines in the config file with dataset names
sed -i "s/VG_SGG_FN.*/VG_SGG_FN = stanford_path('$VG_SGG_FN')/" config.py
sed -i "s/VG_SGG_DICT_FN.*/VG_SGG_DICT_FN = stanford_path('$VG_SGG_DICT_FN')/" config.py
mkdir -p checkpoints/${CHECKPOINT_NAME}
touch checkpoints/${CHECKPOINT_NAME}/train.log
time python models/train_rels.py \
    -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -save_dir checkpoints/$CHECKPOINT_NAME \
    -nepoch 50 -use_bias \
    |& tee checkpoints/${CHECKPOINT_NAME}/train.log

# ---------- 

CHECKPOINT_NAME=0512-oracle
DATA_PREFIX=10R-oracle
echo "TRAINING MOTIFNET ${CHECKPOINT_NAME}"
VG_SGG_FN=${DATA_PREFIX}-VG-SG.h5
VG_SGG_DICT_FN=${DATA_PREFIX}-VG-SG-dicts.json
#HACK: replace lines in the config file with dataset names
sed -i "s/VG_SGG_FN.*/VG_SGG_FN = stanford_path('$VG_SGG_FN')/" config.py
sed -i "s/VG_SGG_DICT_FN.*/VG_SGG_DICT_FN = stanford_path('$VG_SGG_DICT_FN')/" config.py
mkdir -p checkpoints/${CHECKPOINT_NAME}
touch checkpoints/${CHECKPOINT_NAME}/train.log
time python models/train_rels.py \
    -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -save_dir checkpoints/$CHECKPOINT_NAME \
    -nepoch 50 -use_bias \
    |& tee checkpoints/${CHECKPOINT_NAME}/train.log

