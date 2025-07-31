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


def plot_scores(events,
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

    # loop over pairs of categories
    for signal_category_name, signal_category_settings in signal_categories.items():
        for background_category_name, background_category_settings in background_categories.items():

                # get scores for signal and background
                scores_sig = scores[masks[signal_category_name]]
                weights_sig = weights[masks[signal_category_name]]
                scores_bkg = scores[masks[background_category_name]]
                weights_bkg = weights[masks[background_category_name]]

                # safety for zero passing events
                if len(scores_sig)==0 or len(scores_bkg)==0: continue

                # calculate AUC
                # note: the function below cannot handle negative weights,
                #       so take absolute value)
                this_scores = np.concatenate((scores_sig, scores_bkg))
                this_weights = np.concatenate((weights_sig, weights_bkg))
                this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
                auc = roc_auc_score(this_labels, this_scores, sample_weight=np.abs(this_weights))

                # make histograms
                fig, ax = plt.subplots()
                #bins = np.linspace(np.amin(this_scores), np.amax(this_scores), num=51)
                bins = np.linspace(0, 1, num=51)
                hist_sig = np.histogram(scores_sig, bins=bins, weights=weights_sig)[0]
                norm_sig = np.sum( np.multiply(hist_sig, np.diff(bins) ) )
                staterrors_sig = np.sqrt(np.histogram(scores_sig, bins=bins,
                    weights=np.square(weights_sig))[0])
                ax.stairs(hist_sig/norm_sig, edges=bins,
                  color = signal_category_settings['color'],
                  label = signal_category_settings['label'],
                  linewidth=2)
                ax.stairs((hist_sig+staterrors_sig)/norm_sig,
                        baseline=(hist_sig-staterrors_sig)/norm_sig,
                        color = signal_category_settings['color'],
                        edges=bins, fill=True, alpha=0.15)
                hist_bkg = np.histogram(scores_bkg, bins=bins, weights=weights_bkg)[0]
                norm_bkg = np.sum( np.multiply(hist_bkg, np.diff(bins) ) )
                staterrors_bkg = np.sqrt(np.histogram(scores_bkg, bins=bins,
                    weights=np.square(weights_bkg))[0])
                ax.stairs(hist_bkg/norm_bkg, edges=bins,
                  color = background_category_settings['color'],
                  label = background_category_settings['label'],
                  linewidth=2)
                ax.stairs((hist_bkg+staterrors_bkg)/norm_bkg,
                        baseline=(hist_bkg-staterrors_bkg)/norm_bkg,
                        color = background_category_settings['color'],
                        edges=bins, fill=True, alpha=0.15)

                # plot aesthetics
                ylim_default = ax.get_ylim()
                ax.set_ylim((0., ylim_default[1]*1.3))
                ax.grid(which='both')
                ax.set_xlabel('Classifier output score', fontsize=12)
                ax.set_ylabel('Events (normalized)', fontsize=12)
                ax.set_title(f'Score distribution', fontsize=12)
                txt = ax.text(0.95, 0.95, 'AUC: {:.2f}'.format(auc), fontsize=12,
                        ha='right', va='top', transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='white', alpha=0.8))
                leg = ax.legend(fontsize=10)
                for lh in leg.legend_handles:
                    lh.set_alpha(1)
                    lh._sizes = [30]
                fig.tight_layout()
                figname = os.path.join(outputdir,
                  f'scores_{signal_category_name}_vs_{background_category_name}.png')
                fig.savefig(figname)
                print(f'Saved figure {figname}.')
    
                # same with log scale
                ax.autoscale()
                ax.set_yscale('log')
                fig.tight_layout()
                figname = os.path.join(outputdir,
                  f'scores_{signal_category_name}_vs_{background_category_name}_log.png')
                fig.savefig(figname)
                print(f'Saved figure {figname}.')
                plt.close()


def plot_roc(events,
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

    # loop over pairs of categories
    for signal_category_name, signal_category_settings in signal_categories.items():
        for background_category_name, background_category_settings in background_categories.items():

                # get scores for signal and background
                scores_sig = scores[masks[signal_category_name]]
                weights_sig = weights[masks[signal_category_name]]
                scores_bkg = scores[masks[background_category_name]]
                weights_bkg = weights[masks[background_category_name]]

                # safety for zero passing events
                if len(scores_sig)==0 or len(scores_bkg)==0: continue

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
                fig, ax = plt.subplots()
                label = signal_category_settings['label'] + ' vs. '
                label += background_category_settings['label']
                label += ' (AUC: {:.2f})'.format(auc)
                ax.plot(efficiency_bkg, efficiency_sig,
                  color='dodgerblue', linewidth=3, label=label)
    
                # other plot settings
                ax.plot(efficiency_bkg, efficiency_bkg,
                  color='darkblue', linewidth=3, linestyle='--', label='Baseline')
                ax.grid(which='both')
                ax.set_xlabel('Background pass-through', fontsize=12)
                ax.set_ylabel('Signal efficiency', fontsize=12)
                leg = ax.legend()
                fig.tight_layout()
                figname = os.path.join(outputdir,
                  f'roc_{signal_category_name}_vs_{background_category_name}.png')
                fig.savefig(figname)
                print(f'Saved figure {figname}.')
                plt.close()
