import argparse
import random
import os
import subprocess
import numpy as np
from os.path import basename, splitext

parser = argparse.ArgumentParser(description='Run the model with varying parameters.')
parser.add_argument('--gpu', type=str, default='0',
                    help='Specifies which GPU to use.')
parser.add_argument('--model-prefix', type=str, default='sgcls',
                    help='Specifies model prefix')
parser.add_argument('--nruns', type=int, default=20,
                    help='How many variations should we run?')
args = parser.parse_args()

NM_DIR='/dfs/scratch0/vschen/neural-motifs/'

for _ in range(args.nruns):
    # Specify which parameters you want to sweep across and what function to use to sweep over them.
    # For example, random.uniform will sweep over a range independently.
    params = {
        'lr': '%.8f' % 10 ** random.uniform(-6, -1),
    }

    model_name = os.path.join(args.model_prefix + '_' + '_'.join(
            [k + ':' + str(splitext(basename(params[k]))[0]) if 'data' in k
            else k + str(params[k]) for k in params])) 
    arguments = ' '.join(['-' + k + ' ' + str(params[k]) for k in params])
    train = 'mkdir -p checkpoints/' + model_name + '; ' \
        'CUDA_VISIBLE_DEVICES=' + args.gpu + ' python models/train_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -ngpu 1 -nepoch 50 -use_bias' \
         + ' -ckpt ' + NM_DIR + 'checkpoints/vgdet/vg-24.tar' \
         + ' ' + arguments + ' ' \
         + '-save_dir checkpoints/' + model_name \
         + ' |& tee checkpoints/' + model_name + '/train.log'
    print ('^'*89)
    print (train)
    subprocess.call(train, shell=True, executable='/bin/bash')
