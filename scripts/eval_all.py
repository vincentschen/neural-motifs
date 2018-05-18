'''evaluates the specified list of models'''

import argparse
import os
import subprocess

def replace_dataset_command(dataset_prefix):
    h5_path = dataset_prefix + '-VG-SG.h5'
    json_path = dataset_prefix + '-VG-SG-dicts.json'
    replace_h5 = 'sed -i "s/VG_SGG_FN.*/VG_SGG_FN = stanford_path(\'' + h5_path +  '\')/" config.py' 
    replace_json = 'sed -i "s/VG_SGG_DICT_FN.*/VG_SGG_DICT_FN = stanford_path(\''+ json_path + '\')/" config.py'
    return replace_h5 + '; ' + replace_json 

def get_relation_dataset_prefixes():
    relations = ['carry', 'ride', 'eat', 'sit', 'park', 'lay', 'walk', 'hang', 'cover']
    prefixes = []
    for r in relations: 
        prefixes.append('test-' + r) 
    return prefixes 

def add_logging_to_command(command, model, mode):
    track = 'echo "----EVALING %s %s----";' % (model, mode)
    log = ' |& tee checkpoints/' + model + '/evaluate_' + mode + '.log' 
    return track + command + log 

def main(args):
    # ensure correct dataset
    set_full_dataset_command = replace_dataset_command(args.full_dataset_prefix)

    # eval on latest checkpoint 
    checkpoints = os.listdir('%s/%s' % (args.model_dir, args.model_path))
    checkpoints = ['%s/%s/%s' % (args.model_dir, args.model_path, ckpt) for ckpt in checkpoints if 'vgrel' in ckpt] 
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    sgcls_command = 'python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -nepoch 50 -use_bias -test -ckpt %s' % (latest_checkpoint)
    predcls_command = 'python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -nepoch 50 -use_bias -test -ckpt %s' % (latest_checkpoint)

    # combine commands
    sgcls = add_logging_to_command(sgcls_command, args.model_path, 'full-sg')
    pred = add_logging_to_command(predcls_command, args.model_path, 'full-pred')
    command = set_full_dataset_command + '; ' + sgcls + '; ' + pred

    # get relation-level commands
    if args.relation_level_scores: 
        relation_prefixes = get_relation_dataset_prefixes() 
        rel_commands = []
        for p in relation_prefixes: 
            set_rel_dataset_command = replace_dataset_command(p)
            sgcls = add_logging_to_command(sgcls_command, args.model_path, p + '-sg')
            pred = add_logging_to_command(predcls_command, args.model_path, p + '-pred')
            rel_command = set_rel_dataset_command + '; ' + sgcls + '; ' + pred
            rel_commands.append(rel_command)
        rel_commands = "; ".join(rel_commands)

        # append rel-based commands to full_commands
        command = command + '; ' + rel_commands
        
    # call commands
    print ('^'*89)
    print ('EVALING FOR %s' % args.model_path)
    print (command)
    subprocess.call(command, shell=True, executable='/bin/bash')

if __name__=='__main__': 
    parser = argparse.ArgumentParser(description='Run the model with varying parameters.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Specifies which GPU to use.')
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                        help='Specifies which GPU to use.')
    parser.add_argument('--model_path', type=str, default='test',
                        help='Specifies the model we should evaluate')
    parser.add_argument('--relation_level_scores', action='store_true' , default=False,
                        help='Whether evaluate at the class level.')
    parser.add_argument('--full_dataset_prefix', type=str, default='10R-easy-hybrid-1k', 
                        help='prefix for full dataset')
    args = parser.parse_args()
    main(args)
    
