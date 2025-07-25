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
    xsecweighting = False
    signal_categories = {
        'HH': {
            'branch': 'isSignal',
            'color': 'red',
            'label': r'HH $\rightarrow$ 4b'
        }
    }
    background_categories = {
        'syndata': {
            'branch': 'isBackground',
            'color': 'blue',
            'label': 'Syn. data'
        }
    }
    all_categories = {**signal_categories, **background_categories}
    correlation_variables = {
        'mH1': {
            'branch': 'dHH_H1_mass',
            'label': '$m(H_{1})$',
        },
        #'mH2': {
        #    'branch': 'dHH_H2_mass',
        #    'label': '$m(H_{2})$ [GeV]',
        #},
        #'mHH': {
        #    'branch': 'dHH_HH_mass',
        #    'label': '$m(HH)$ [GeV]',
        #},
        #'mHavg': {
        #    'branch': 'hh_average_mass',
        #    'label': '$m(H_{avg}) [GeV]$',
        #}
    }

    # find all branches to read
    branches_to_read = (
        [score_branch]
        + [cat['branch'] for cat in all_categories.values()]
        + [v['branch'] for v in correlation_variables.values()]
    )
    if xsecweighting:
        branches_to_read += ['lumiwgt', 'genWeight', 'xsecWeight']

    # handle case of provided input json file
    if args.resultdir.endswith('.json'):
        with open(args.resultdir, 'r') as f:
            results = json.load(f)

    # handle case of provided result directory
    else:
    
      # loop over alpha directories inside the result directory
      alphadirs = ([d for d in os.listdir(args.resultdir)
                    if os.path.isdir(os.path.join(args.resultdir, d))])
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

            # read events
            events = get_events_from_file(resultfile,
                       treename = treename,
                       branches = branches_to_read)

            # get scores
            scores = events[score_branch]
            
            # get weights
            weights = np.ones(len(scores))
            if xsecweighting:
                weights = np.multiply(events['lumiwgt'],
                            np.multiply(events['genWeight'], events['xsecWeight']))

            # get mask for each category
            masks = {}
            for category_name, category_settings in all_categories.items():
                branch = category_settings['branch']
                mask = events[branch].astype(bool)
                masks[category_name] = mask

            # loop over signal and background categories for AUC calculation
            aucs = {}
            for sig_name, sig_settings in signal_categories.items():
                for bkg_name, bkg_settings in background_categories.items():

                    # get scores for signal and background
                    scores_sig = scores[masks[sig_name]]
                    weights_sig = weights[masks[sig_name]]
                    scores_bkg = scores[masks[bkg_name]]
                    weights_bkg = weights[masks[bkg_name]]
                    
                    # calculate AUC
                    this_scores = np.concatenate((scores_sig, scores_bkg))
                    this_weights = np.concatenate((weights_sig, weights_bkg))
                    this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
                    auc = roc_auc_score(this_labels, this_scores, sample_weight=np.abs(this_weights))
                    aucs[f'{sig_name}_vs_{bkg_name}'] = auc

            # append auc results to this alpha entry
            result['aucs_raw'].append(aucs)

            # loop over categories for DisCo calculation
            dccoeffs = {}
            for cat_name, cat_settings in all_categories.items():

                # calculate distance correlation
                thisdccoeffs = get_discos_from_events(events,
                                 score_branch = score_branch,
                                 correlation_variables = correlation_variables,
                                 mask_branch = cat_settings['branch'],
                                 npoints=1000, niterations=10)
                dccoeffs[cat_name] = thisdccoeffs

                # alternatively, use a simpler/faster correlation metric
                #dccoeffs[cat_name] = {}
                #this_scores = scores[masks[cat_name]]
                #for var_name, var_settings in correlation_variables.items():
                #    var_values = events[var_settings['branch']][masks[cat_name]]
                #    corrcoef = abs(np.corrcoef(this_scores, var_values)[0,1])
                #    dccoeffs[cat_name][var_name] = corrcoef

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
        for category in all_categories:
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
    sorted_ids = np.argsort(alphas)
    alphas = alphas[sorted_ids]
    aucs = {}
    for sig_name, sig_settings in signal_categories.items():
        for bkg_name, bkg_settings in background_categories.items():
            key = f'{sig_name}_vs_{bkg_name}'
            auc_avg = np.array([results[idx][f'auc_avg_{key}'] for idx in sorted_ids])
            auc_min = np.array([results[idx][f'auc_min_{key}'] for idx in sorted_ids])
            auc_max = np.array([results[idx][f'auc_max_{key}'] for idx in sorted_ids])
            aucs[key] = {'avg': auc_avg, 'min': auc_min, 'max': auc_max}
    dccoeffs = {}
    for cat_name, cat_settings in all_categories.items():
        for var_name, var_settings in correlation_variables.items():
            key = f'{cat_name}_{var_name}'
            dccoeff_avg = np.array([results[idx][f'dccoeff_avg_{key}'] for idx in sorted_ids])
            dccoeff_min = np.array([results[idx][f'dccoeff_min_{key}'] for idx in sorted_ids])
            dccoeff_max = np.array([results[idx][f'dccoeff_max_{key}'] for idx in sorted_ids])
            dccoeffs[key] = {'avg': dccoeff_avg, 'min': dccoeff_min, 'max': dccoeff_max}
    sorted_ids = np.argsort(alphas)

    # make a plot
    print('Making plot of AUCs...')
    fig, axs = plt.subplots(figsize=(12,6), ncols=2)
    ax = axs[0]
    ncolors = len(signal_categories)*len(background_categories)
    cmap = plt.get_cmap('cool', ncolors)
    cidx = 0
    # base plot
    for sig_name, sig_settings in signal_categories.items():
        for bkg_name, bkg_settings in background_categories.items():
            key = f'{sig_name}_vs_{bkg_name}'
            label = f'{sig_settings["label"]} vs. {bkg_settings["label"]}'
            color = cmap(cidx); cidx += 1
            ax.plot(alphas, aucs[key]['avg'], color=color, label=label)
            ax.fill_between(alphas, aucs[key]['min'], aucs[key]['max'], color=color, alpha=0.5)
    # aesthetics
    ax.legend(fontsize=12)
    ax.set_xlabel('Disco strength parameter', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim((0.5, 1.))
    ax.grid(visible=True, which='both')

    print('Making plot of correlations...')
    ax = axs[1]
    ncolors = len(all_categories)*len(correlation_variables)
    cmap = plt.get_cmap('cool', ncolors)
    cidx = 0
    # base plot
    for cat_name, cat_settings in all_categories.items():
        for var_name, var_settings in correlation_variables.items():
            key = f'{cat_name}_{var_name}'
            label = f'{cat_settings["label"]} ({var_settings["label"]})'
            color = cmap(cidx); cidx += 1
            ax.plot(alphas, dccoeffs[key]['avg'], linestyle='--', color=color, label=label)
            ax.fill_between(alphas, dccoeffs[key]['min'], dccoeffs[key]['max'], color=color, alpha=0.5)
    # aesthetics
    ax.legend(fontsize=12)
    ax.set_xlabel('Disco strength parameter', fontsize=12)
    ax.set_ylabel('Correlation strength', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim((0.001, 1.))
    ax.grid(visible=True, which='both')

    # save figure
    fig.tight_layout()
    fig.savefig(args.outputfile)
