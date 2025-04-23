import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(weavercoredir)
from weaver.studies.evaluation.plot_history_multi import plot_history_multi


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resultdir', required=True)
    parser.add_argument('-o', '--outputfile', required=True)
    args = parser.parse_args()

    # hard-coded settings
    filename = 'network_history_train.json'
    
    # loop over alpha directories inside the result directory
    alphadirs = ([d for d in os.listdir(args.resultdir)
                    if os.path.isdir(os.path.join(args.resultdir, d))])
    print(f'Found {len(alphadirs)} alpha directories.')
    histories = {}
    alphas = []
    for alphadir in sorted(alphadirs):

        # find alpha value from directory name
        alpha = float(alphadir.split('_')[-1].replace('p', '.'))
        alphas.append(alpha)

        # get first subdirectory (consider only one of parallel trainings)
        alphadir = os.path.join(args.resultdir, alphadir)
        for subdir in sorted(os.listdir(alphadir))[:1]:
            subdir = os.path.join(alphadir, subdir)
            print('Checking {}...'.format(subdir), end='\r')
            
            # find result file
            resultfile = os.path.join(subdir, filename)
            if not os.path.exists(resultfile):
                print('WARNING: {} does not exist, skipping.'.format(resultfile))
                continue

            # read history
            with open(resultfile, 'r') as f:
                history = json.load(f)
            histories[r'$\alpha$ = ' + '{:.2f}'.format(alpha)] = history
    print('')

    # re-order the dict according to numerical value of alpha
    alphas = np.array(alphas)
    sorted_indices = np.argsort(alphas)
    keys = list(histories.keys())
    new_histories = {}
    for idx in sorted_indices:
        new_histories[keys[idx]] = histories[keys[idx]]
    histories = new_histories

    # make plot
    fig, axs = plot_history_multi(histories)

    # save figure
    fig.savefig(args.outputfile)
    print(f'Saved figure {args.outputfile}')
