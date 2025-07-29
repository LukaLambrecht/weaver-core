# Plot score distribution and ROC for multiple processes

import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from evaluationtools import get_events_from_file


def plot_scores_multi(events,
            categories,
            xsecweighting = False,
            outputdir = None,
            score_branch = None,
            variable = None):

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    
    # get scores
    scores = events[score_branch]
    if len(scores)==0: return

    # get weights
    weights = np.ones(len(scores))
    if xsecweighting:
        weights = np.multiply(events['lumiwgt'], np.multiply(events['genWeight'], events['xsecWeight']))

    # get mask for each category
    cat_masks = {}
    for category_name, category_settings in categories.items():
        branch = category_settings['branch']
        mask = events[branch].astype(bool)
        cat_masks[category_name] = mask

    # create dummy secondary variable if none was provided
    # (for optional splitting of the score distribution for each process
    # into additional bins based on a provided variable)
    if variable is None:
        variable = {
            'branch': score_branch,
            'label': '',
            'bins': [-99, 99]
        }

    # make mask for each secondary variable bin
    # loop over bins in secondary variable
    var_masks = []
    var_values = var_values = events[variable['branch']]
    for idx in range(len(variable['bins'])-1):
        minvarvalue = variable['bins'][idx]
        maxvarvalue = variable['bins'][idx+1]
        mask = ((var_values >= minvarvalue) & (var_values <= maxvarvalue))
        var_masks.append(mask)

    # set line styles for different secondary variable bins
    linestyles = ['solid']
    for idx in range(len(variable['bins'])-1):
        linestyles.append( (0, (1, 1 + 1*idx)) )

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # initialize figure
    fig, ax = plt.subplots()

    # loop over categories
    for category_name, category_settings in categories.items():
        cat_mask = cat_masks[category_name]

        # loop over bins in secondary variable
        nids = len(variable['bins'])
        if nids<=2: nids = 1
        # (if only one bin (or invalid) was provided,
        #  only plot total score, not binned in secondary variable)
        for idx in range(nids):
                # first iteration is all (no splitting)
                if idx==0:
                    var_mask = np.ones(len(cat_mask)).astype(bool)
                    var_bin_label = ''
                # otherwise split in secondary bins
                else:
                    var_mask = var_masks[idx-1]
                    var_bin_label = ' ({}: {:.2f} - {:.2f})'.format(
                                  variable['label'],
                                  variable['bins'][idx-1],
                                  variable['bins'][idx])

                # get scores
                total_mask = (cat_mask & var_mask)
                this_values = scores[total_mask]
                this_weights = weights[total_mask]
            
                # make a histogram
                label = category_settings['label']
                if len(variable['bins'])>2: label += ' ' + var_bin_label
                #bins = np.linspace(np.amin(scores), np.amax(scores), num=31)
                bins = np.linspace(0, 1, num=41)
                hist = np.histogram(this_values, bins=bins, weights=this_weights)[0]
                norm = np.sum( np.multiply(hist, np.diff(bins) ) )
                staterrors = np.sqrt(np.histogram(this_values, bins=bins, weights=np.square(this_weights))[0])
                ax.stairs(hist/norm, edges=bins,
                  color = category_settings['color'],
                  label = label,
                  linestyle = linestyles[idx],
                  linewidth=2)
                ax.stairs((hist+staterrors)/norm, baseline=(hist-staterrors)/norm,
                        color = category_settings['color'],
                        edges=bins, fill=True, alpha=0.15)
        
    ax.set_xlabel('Classifier output score', fontsize=12)
    ax.set_ylabel('Events (normalized)', fontsize=12)
    ax.set_title(f'Score distribution', fontsize=12)
    leg = ax.legend(fontsize=10)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
        lh._sizes = [30]
    fig.tight_layout()
    figname = os.path.join(outputdir, f'scores_multi.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
    
    # same with log scale
    ax.set_yscale('log')
    fig.tight_layout()
    figname = os.path.join(outputdir, f'scores_multi_log.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
    plt.close()


def plot_roc_multi(events,
            signal_categories,
            background_categories,
            xsecweighting = False,
            outputdir = None,
            score_branch = None):

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    all_categories = {**signal_categories, **background_categories}

    # get scores
    scores = events[score_branch]

    # get weights
    weights = np.ones(len(scores))
    if xsecweighting:
        weights = np.multiply(events['lumiwgt'], np.multiply(events['genWeight'], events['xsecWeight']))

    # get mask for each category
    masks = {}
    for category_name, category_settings in all_categories.items():
        branch = category_settings['branch']
        mask = events[branch].astype(bool)
        masks[category_name] = mask

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # initialize figure
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('cool', len(signal_categories)*len(background_categories))
    cidx = 0

    # loop over pairs of categories
    for signal_category_name, signal_category_settings in signal_categories.items():
        for background_category_name, background_category_settings in background_categories.items():

                # get scores for signal and background
                scores_sig = scores[masks[signal_category_name]]
                weights_sig = weights[masks[signal_category_name]]
                scores_bkg = scores[masks[background_category_name]]
                weights_bkg = weights[masks[background_category_name]]
                
                # safety for no passing events
                if len(scores_sig)==0 or len(scores_bkg)==0:
                    continue

                # calculate AUC
                # note: the function below cannot handle negative weights,
                #       so take absolute value)
                this_scores = np.concatenate((scores_sig, scores_bkg))
                this_weights = np.concatenate((weights_sig, weights_bkg))
                this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
                auc = roc_auc_score(this_labels, this_scores, sample_weight=np.abs(this_weights))

                # calculate signal and background efficiency
                thresholds = np.linspace(np.amin(this_scores), np.amax(this_scores), num=100)
                efficiency_sig = np.zeros(len(thresholds))
                efficiency_bkg = np.zeros(len(thresholds))
                for idx, threshold in enumerate(thresholds):
                    eff_s = np.sum(weights_sig[scores_sig > threshold])
                    efficiency_sig[idx] = eff_s
                    eff_b = np.sum(weights_bkg[scores_bkg > threshold])
                    efficiency_bkg[idx] = eff_b
                efficiency_sig /= np.sum(weights_sig)
                efficiency_bkg /= np.sum(weights_bkg)

                # make a plot of the ROC curve
                label = signal_category_settings['label'] + ' vs. '
                label += background_category_settings['label']
                label += ' (AUC: {:.2f})'.format(auc)
                ax.plot(efficiency_bkg, efficiency_sig,
                  color=cmap(cidx), linewidth=3, label=label)
                cidx += 1
    
    # other plot settings
    dummy_efficiency = np.linspace(0, 1, num=101)
    ax.plot(dummy_efficiency, dummy_efficiency,
      color='darkblue', linewidth=3, linestyle='--', label='Baseline')
    ax.set_xlabel('Background pass-through', fontsize=12)
    ax.set_ylabel('Signal efficiency', fontsize=12)
    leg = ax.legend()
    fig.tight_layout()
    figname = os.path.join(outputdir, 'roc_multi.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
    plt.close()
