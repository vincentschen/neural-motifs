import argparse
import json
import numpy as np
import os

def get_model_score(dirname, field_name):
    """Returns the final score achieved by the model in the directory."""
    log = os.path.join(dirname, 'metrics.json')
    try:
        metrics = json.load(open(log))
    except:
        return -1000, None
    return metrics[field_name], {k:metrics[k] for k in metrics}


def save_metrics(dirname): 
    """Given a directory, saves out 'metrics.json' based on last validation score in train.log"""
    scores = {}
    with open(dirname + '/train.log') as f: 
        # read file lines in reverse
        lines = list(f)[::-1] 
        for i in range(len(lines)): 
            # find first instance of scores
            if 'R@20' in lines[i]: 
                try: 
                    scores['r20'] = float(lines[i][6:])
                    scores['r50'] = float(lines[i+1][6:])
                    scores['r100'] = float(lines[i+2][7:]) 
                except Exception as e:
                    print (e) 
                    print ('incomplete or corrupt model: %s' % dirname) 

                break
    outfile = dirname + '/metrics.json'
    print ('saving out metrics for %s' % outfile)
    json.dump(scores, open(outfile, 'w'))  

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='checkpoints',
                        help='The directory with all the models.')
    parser.add_argument('--prefix', type=str, default='',
                        help='The prefix for each model directory we '
                        'are investigating.')
    parser.add_argument('--filter-words', type=str, nargs='+',
                        default=[],
                        help='Filter words in the directory name.')
    parser.add_argument('--negative-words', type=str, nargs='+',
                        default=[],
                        help='Words that should not be in the directory name.')
    parser.add_argument('--eval-field-name', type=str, default='r100',
                        help='Filter words in the directory name.')
    parser.add_argument('--n-models', type=int, default=1,
                        help='Number of top n models to show.')
    args = parser.parse_args()

    print ("Scanning %d model[s] that start with %s, have [%s] in name is, and exclude [%s] in name..." % (
           args.n_models, args.prefix, ', '.join(args.filter_words), ','.join(args.negative_words)))

    all_models = os.listdir(args.dir)

    scores, evals, names = [], [], []
    for name in all_models:
        if not name.startswith(args.prefix):
            continue
        if len(args.filter_words) > 0:
            found = True
            for word in args.filter_words:
                if word not in name:
                    found = False
                    break
            if not found:
                continue
        if len(args.negative_words) > 0:
            found = False
            for word in args.negative_words:
                if word in name:
                    found = True
                    break
            if found:
                continue

        dirname = os.path.join(args.dir, name)
        checkpoint_files = os.listdir(dirname)

        if 'train.log' not in checkpoint_files: 
            continue 

        if 'metrics.json' not in checkpoint_files: 
            save_metrics(dirname)

        score, all_evals = get_model_score(dirname, args.eval_field_name)
        scores.append(score)
        evals.append(all_evals)
        names.append(name)
        break
    # sort by metric score
    best_idx = np.argsort(scores)[::-1][:args.n_models]
    for i, idx in enumerate(best_idx):
        print ("[Model %d]: %s" % (i, names[idx]))
        print ("Best %s: %s, Full Eval: %s" % (args.eval_field_name, str(scores[idx]), evals[idx])) 
