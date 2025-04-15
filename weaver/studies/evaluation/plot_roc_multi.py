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
            score_branch = None):

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    
    # get scores
    scores = events[score_branch]

    # get weights
    weights = np.ones(len(scores))
    if xsecweighting:
        weights = np.multiply(events['lumiwgt'], np.multiply(events['genWeight'], events['xsecWeight']))

    # get mask for each category
    masks = {}
    for category_name, category_settings in categories.items():
        branch = category_settings['branch']
        mask = events[branch].astype(bool)
        masks[category_name] = mask

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # initialize figure
    fig, ax = plt.subplots()

    # loop over categories
    for category_name, category_settings in categories.items():

                this_values = scores[masks[category_name]]
                this_weights = weights[masks[category_name]]
            
                # make a histogram
                bins = np.linspace(np.amin(scores), np.amax(scores), num=31)
                hist = np.histogram(this_values, bins=bins, weights=this_weights)[0]
                norm = np.sum( np.multiply(hist, np.diff(bins) ) )
                staterrors = np.sqrt(np.histogram(this_values, bins=bins, weights=np.square(this_weights))[0])
                ax.stairs(hist/norm, edges=bins,
                  color = category_settings['color'],
                  label = category_settings['label'],
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
    ax.plot(efficiency_bkg, efficiency_bkg,
      color='darkblue', linewidth=3, linestyle='--', label='Baseline')
    ax.set_xlabel('Background pass-through', fontsize=12)
    ax.set_ylabel('Signal efficiency', fontsize=12)
    leg = ax.legend()
    figname = os.path.join(outputdir, 'roc_multi.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
    plt.close()
