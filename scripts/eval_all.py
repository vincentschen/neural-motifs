'''evaluates the specified list of models'''

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='Run the model with varying parameters.')
parser.add_argument('--gpu', type=str, default='0',
                    help='Specifies which GPU to use.')
parser.add_argument('--model_dir', type=str, default='checkpoints',
                    help='Specifies which GPU to use.')

args = parser.parse_args()

models = [
    '0517-hybrid-10k_lr0.00002527', 
    '0517-hybrid-10k_lr0.00009314',
    '0517-hybrid-10k_lr0.00028797',
    '0517-hybrid-10k_lr0.00043643',
    '0517-hybrid-10k_lr0.00075724',
    '0517-hybrid-10k_lr0.00093993',
    '0517-hybrid-10k_lr0.00274958'
]

for model in models: 
    checkpoints = os.listdir('%s/%s' % (args.model_dir, model))
    checkpoints = ['%s/%s/%s' % (args.model_dir, model, ckpt) for ckpt in checkpoints if 'vgrel' in ckpt] 
    print (checkpoints)
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print (latest_checkpoint)
    
    sgcls_command = 'python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -nepoch 50 -use_bias -test -ckpt %s' % (latest_checkpoint)
    predcls_command = 'python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -nepoch 50 -use_bias -test -ckpt %s' % (latest_checkpoint)
    command = sgcls_command + '; ' + predcls_command + ' |& tee checkpoints/' + model + '/evaluate.log'
    print ('^'*89)
    print ('EVALING FOR %s' % model)
    print (command)
    subprocess.call(command, shell=True, executable='/bin/bash')
