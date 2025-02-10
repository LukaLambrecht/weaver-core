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
    args = parser.parse_args()

    # hard-coded settings
    filename = 'output.root'
    treename = 'Events'
    score_branch = 'score_isSignal'
    signal_categories = ['isSignal']
    background_categories = ['isQCD', 'isTT']
    correlation_categories = ['isQCD', 'isTT']
    correlation_variables = ['dHH_H1_regmass']

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
        result = {'alpha': 0., 'aucs_raw': [], 'dccoeffs_raw': []}

        # find alpha value from directory name
        alpha = float(alphadir.split('_')[-1].replace('p', '.'))
        result['alpha'] = alpha

        # loop over subdirectories with parallel trainings
        alphadir = os.path.join(args.resultdir, alphadir)
        for subdir in sorted(os.listdir(alphadir)):
            subdir = os.path.join(alphadir, subdir)
            print('Checking {}...'.format(subdir), end='\r')
            
            # find result file
            resultfile = os.path.join(subdir, filename)
            if not os.path.exists(resultfile):
                print('WARNING: {} does not exist, skipping.'.format(resultfile))
                continue

            # read events and scores from input file
            correlation_branches = correlation_categories + correlation_variables
            events = get_events_from_file(resultfile,
                       treename = treename,
                       signal_branches = signal_categories,
                       background_branches = background_categories,
                       correlation_branches = correlation_branches)

            # loop over signal and background categories for AUC calculation
            aucs = {}
            for signal_category in signal_categories:
                for background_category in background_categories:

                    # get scores and labels
                    (scores, labels) = get_scores_from_events(events,
                            score_branch = score_branch,
                            signal_branch = signal_category,
                            background_branch = background_category)

                    # calculate AUC
                    auc = roc_auc_score(labels, scores)
                    aucs[f'{signal_category}_vs_{background_category}'] = auc

            # append auc results to this alpha entry
            result['aucs_raw'].append(aucs)

            # loop over categories for DisCo calculation
            dccoeffs = {}
            for category in correlation_categories:

                # calculate distance correlation
                thisdccoeffs = get_discos_from_events(events,
                                 score_branch = score_branch,
                                 variable_branches = correlation_variables,
                                 mask_branch = category,
                                 npoints=1000, niterations=5)
                dccoeffs[category] = thisdccoeffs

            # add disco results to this alpha entry
            result['dccoeffs_raw'].append(dccoeffs)

        # safety for empty arrays
        if len(result['aucs_raw']) == 0:
            print(f'WARNING: no results found for alpha = {alpha}, skipping...')
            continue

        # calculate mean and std
        for signal_category in signal_categories:
            for background_category in background_categories:
                key = f'{signal_category}_vs_{background_category}'
                avg = np.mean([el[key] for el in result['aucs_raw']])
                std = np.std([el[key] for el in result['aucs_raw']])
                result[f'auc_avg_{key}'] = avg
                result[f'auc_min_{key}'] = avg - std
                result[f'auc_max_{key}'] = avg + std
        for category in correlation_categories:
            for variable in correlation_variables:
                key = f'{category}_{variable}'
                avg = np.mean([el[category][variable] for el in result['dccoeffs_raw']])
                std = np.std([el[category][variable] for el in result['dccoeffs_raw']])
                result[f'dccoeff_avg_{key}'] = avg
                result[f'dccoeff_min_{key}'] = avg - std
                result[f'dccoeff_max_{key}'] = avg + std
        results.append(result)
      print()

      # write to file for later quicker retrieval
      resultfile = os.path.splitext(args.outputfile)[0]+ '.json'
      with open(resultfile, 'w') as f:
          json.dump(results, f)

    # format results
    print('Formatting results...')
    alphas = np.array([results[idx]['alpha'] for idx in range(len(results))])
    aucs = {}
    for signal_category in signal_categories:
        for background_category in background_categories:
            key = f'{signal_category}_vs_{background_category}'
            auc_avg = np.array([results[idx][f'auc_avg_{key}'] for idx in range(len(results))])
            auc_min = np.array([results[idx][f'auc_min_{key}'] for idx in range(len(results))])
            auc_max = np.array([results[idx][f'auc_max_{key}'] for idx in range(len(results))])
            aucs[key] = {'avg': auc_avg, 'min': auc_min, 'max': auc_max}
    dccoeffs = {}
    for category in correlation_categories:
        for variable in correlation_variables:
            key = f'{category}_{variable}'
            dccoeff_avg = np.array([results[idx][f'dccoeff_avg_{key}'] for idx in range(len(results))])
            dccoeff_min = np.array([results[idx][f'dccoeff_min_{key}'] for idx in range(len(results))])
            dccoeff_max = np.array([results[idx][f'dccoeff_max_{key}'] for idx in range(len(results))])
            dccoeffs[key] = {'avg': dccoeff_avg, 'min': dccoeff_min, 'max': dccoeff_max}
    sorted_ids = np.argsort(alphas)

    # make a plot
    print('Making plot...')
    fig, ax = plt.subplots()
    ncolors = len(signal_categories)*len(background_categories)
    cmap = plt.get_cmap('rainbow', ncolors)
    cidx = 0
    for signal_category in signal_categories:
        for background_category in background_categories:
            key = f'{signal_category}_vs_{background_category}'
            color = cmap(cidx); cidx += 1
            ax.plot(alphas, aucs[key]['avg'], color=color, label=f'AUC ({key})')
            ax.fill_between(alphas, aucs[key]['min'], aucs[key]['max'], color=color, alpha=0.5)

    ncolors = len(correlation_categories)*len(correlation_variables)
    cmap = plt.get_cmap('rainbow', ncolors)
    cidx = 0
    for category in correlation_categories:
        for variable in correlation_variables:
            key = f'{category}_{variable}'
            color = cmap(cidx); cidx += 1
            ax.plot(alphas, dccoeffs[key]['avg'], linestyle='--', color=color, label=f'DisCo ({key})')
            ax.fill_between(alphas, dccoeffs[key]['min'], dccoeffs[key]['max'], color=color, alpha=0.5)

    ax.legend(fontsize=12)
    ax.set_xlabel('Disco strength parameter', fontsize=12)

    ax.set_yscale('log')
    ax.set_ylim((0.001, 1.))
    ax.grid(visible=True, which='both')

    fig.tight_layout()
    fig.savefig(args.outputfile)
