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
from weaver.studies.evaluation.evaluationtools import get_events_from_file
from weaver.studies.evaluation.evaluationtools import get_scores_from_events
from weaver.studies.evaluation.evaluationtools import get_discos_from_events


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resultdir', required=True)
    parser.add_argument('-o', '--outputfile', required=True)
    parser.add_argument('-s', '--signal_mask', required=True)
    parser.add_argument('-c', '--correlation', default=None)
    parser.add_argument('--plot_roc_dirs', default=[], nargs='+')
    args = parser.parse_args()

    # handle case of provided input json file
    if args.resultdir.endswith('.json'):
        with open(args.resultdir, 'r') as f:
            results = json.load(f)

    # handle case of provided result directory
    else:
    
      # loop over alpha directories inside the result directory
      alphadirs = [d for d in os.listdir(args.resultdir) if os.path.isdir(os.path.join(args.resultdir, d))]
      print(f'Found {len(alphadirs)} alpha directories.')
      results = []
      for alphadir in sorted(alphadirs):
        result = {'alpha': 0., 'aucs': [], 'dccoeffs': []}

        # find alpha value from directory name
        alpha = float(alphadir.split('_')[-1].replace('p', '.'))
        result['alpha'] = alpha

        # loop over subdirectories with parallel trainings
        alphadir = os.path.join(args.resultdir, alphadir)
        for subdir in sorted(os.listdir(alphadir)):
            subdir = os.path.join(alphadir, subdir)
            print('Checking {}...'.format(subdir), end='\r')
            
            # find result file
            resultfile = os.path.join(subdir, 'output.root')
            if not os.path.exists(resultfile):
                print('WARNING: {} does not exist, skipping.'.format(resultfile))
                continue

            # read events and scores from input file
            events = get_events_from_file(resultfile, args.signal_mask, treename='Events',
               correlations=[args.correlation] if args.correlation is not None else None)
            (scores, label) = get_scores_from_events(events, args.signal_mask)

            # calculate AUC
            auc = roc_auc_score(labels, scores)

            # calculate distance correlation
            dccoeff = 0
            if args.correlation is not None:
                mask = (labels == 0) # only for background events
                dccoeffs = get_discos_from_events(events, scores, [args.correlation],
                             npoints=1000, niterations=5, mask=mask)
                dccoeff = dccoeffs[args.correlation]

            # add results
            result['aucs'].append(auc)
            result['dccoeffs'].append(dccoeff)

        # safety for empty arrays
        if len(result['aucs']) == 0:
            print(f'WARNING: no results found for alpha = {alpha}, skipping...')
            continue

        # calculate mean and std
        result['auc_avg'] = np.mean(result['aucs'])
        result['auc_min'] = result['auc_avg'] - np.std(result['aucs'])
        result['auc_max'] = result['auc_avg'] + np.std(result['aucs'])
        result['dccoeff_avg'] = np.mean(result['dccoeffs'])
        result['dccoeff_min'] = result['dccoeff_avg'] - np.std(result['dccoeffs'])
        result['dccoeff_max'] = result['dccoeff_avg'] + np.std(result['dccoeffs'])
        results.append(result)
      print()

      # write to file for later quicker retrieval
      resultfile = os.path.splitext(args.outputfile)[0]+ '.json'
      with open(resultfile, 'w') as f:
          json.dump(results, f)

    # format results
    print('Formatting results...')
    alphas = np.array([results[idx]['alpha'] for idx in range(len(results))])
    auc_avg = np.array([results[idx]['auc_avg'] for idx in range(len(results))])
    auc_min = np.array([results[idx]['auc_min'] for idx in range(len(results))])
    auc_max = np.array([results[idx]['auc_max'] for idx in range(len(results))])
    dccoeff_avg = np.array([results[idx]['dccoeff_avg'] for idx in range(len(results))])
    dccoeff_min = np.array([results[idx]['dccoeff_min'] for idx in range(len(results))])
    dccoeff_max = np.array([results[idx]['dccoeff_max'] for idx in range(len(results))])
    sorted_ids = np.argsort(alphas)
    alphas = alphas[sorted_ids]
    auc_avg = auc_avg[sorted_ids]
    auc_min = auc_min[sorted_ids]
    auc_max = auc_max[sorted_ids]
    dccoeff_avg = dccoeff_avg[sorted_ids]
    dccoeff_min = dccoeff_min[sorted_ids]
    dccoeff_max = dccoeff_max[sorted_ids]

    # make a plot
    print('Making plot...')
    fig, ax = plt.subplots()
    ax.plot(alphas, auc_avg, color='b', label='AUC (classification)')
    ax.fill_between(alphas, auc_min, auc_max, color='b', alpha=0.5)

    ax.plot(alphas, dccoeff_avg, color='r', label='DisCo (correlation)')
    ax.fill_between(alphas, dccoeff_min, dccoeff_max, color='r', alpha=0.5)

    ax.legend(fontsize=12)
    ax.set_xlabel('Disco strength parameter', fontsize=12)

    ax.set_yscale('log')
    ax.set_ylim((0.001, 1.))
    ax.grid(visible=True, which='both')

    fig.tight_layout()
    fig.savefig(args.outputfile)

    # plot full results (roc, correlation, etc)
    # for a selected number of trainings
    if len(args.plot_roc_dirs) > 0:
        plot_roc_dirs = [os.path.abspath(d) for d in args.plot_roc_dirs]
        cmd = 'python ../evaluation/evaluate_loop.py'
        cmd += ' -m {}'.format(' '.join(plot_roc_dirs))
        cmd += ' -i output.root'
        cmd += f' -s {args.signal_mask}'
        cmd += ' -t Events'
        cmd += f' -c {args.correlation}'
        cmd += ' --plot_score_dist'
        cmd += ' --plot_roc'
        cmd += ' --plot_correlation'
        os.system(cmd)
