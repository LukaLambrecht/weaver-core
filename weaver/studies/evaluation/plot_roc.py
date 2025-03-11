import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from evaluationtools import get_scores_from_events
from evaluationtools import get_events_from_file


def plot_roc_from_file(inputfile, treename=None,
        signal_branch=None, background_branch=None,
        xsecweighting=False, **kwargs):
    ### calculate and plot ROC directly from an input file

    # format the signal and background branch names
    signal_branches = [signal_branch] if signal_branch is not None else []
    background_branches = [background_branch] if background_branch is not None else []

    # format weight branch names
    weight_branches = ['genWeight', 'xsecWeight'] if xsecweighting else None

    # read events
    events = get_events_from_file(inputfile, treename=treename,
               signal_branches=signal_branches, background_branches=background_branches,
               weight_branches=weight_branches)

    # plot ROC from events
    return plot_roc_from_events(events, signal_branch=signal_branch,
            background_branch=background_branch,
            xsecweighting=xsecweighting, **kwargs)


def plot_roc_from_events(events,
        score_branch=None, signal_branch=None, background_branch=None,
        xsecweighting=False, **kwargs):
    ### calculate and plot ROC from an events array

    # format the score, signal, and background branch names
    if score_branch is None:
        raise Exception('A score branch must be specified.')
    if signal_branch is None and background_branch is None:
        raise Exception('A signal branch or a background branch (or both) must be specified.')

    # get scores and labels
    (scores, labels, weights) = get_scores_from_events(events,
                                  score_branch=score_branch,
                                  signal_branch=signal_branch,
                                  background_branch=background_branch,
                                  xsecweighting=xsecweighting)

    # plot ROC from scores
    outputtag = f'{signal_branch}_vs_{background_branch}'
    return plot_roc_from_scores(scores, labels, weights=weights, outputtag=outputtag, **kwargs)


def plot_roc_from_scores(scores, labels,
             weights=None,
             outputdir=None,
             outputtag=None,
             plot_score_dist=False,
             plot_roc=False):

    # separate scores into signal and background scores
    scores_sig = scores[labels==1]
    scores_bkg = scores[labels==0]
    if weights is None: weights = np.ones(len(scores))
    weights_sig = weights[labels==1]
    weights_bkg = weights[labels==0]

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # calculate AUC
    print('Calculating AUC (ROC)...')
    # note: the function below cannot handle negative weights,
    #       so take absolute value)
    auc = roc_auc_score(labels, scores, sample_weight=np.abs(weights))
    print('AUC (ROC) on testing set: {:.3f}'.format(auc))

    # make a plot of the score distribution
    if plot_score_dist and outputdir is not None:
        print('Making output score distribution...')
        fig, ax = plt.subplots()
        bins = np.linspace(np.amin(scores_bkg), np.amax(scores_sig), num=50)
        # make histogram for bkg
        hist_bkg = np.histogram(scores_bkg, bins=bins, weights=weights_bkg)[0]
        norm_bkg = np.sum( np.multiply(hist_bkg, np.diff(bins) ) )
        staterrors_bkg = np.sqrt(np.histogram(scores_bkg, bins=bins, weights=np.square(weights_bkg))[0])
        ax.stairs(hist_bkg/norm_bkg, edges=bins,
                  color='dodgerblue', label='Background', linewidth=3)
        ax.stairs((hist_bkg+staterrors_bkg)/norm_bkg, baseline=(hist_bkg-staterrors_bkg)/norm_bkg,
                    color='dodgerblue', edges=bins, fill=True, alpha=0.3)
        # make histogram for sig
        hist_sig = np.histogram(scores_sig, bins=bins, weights=weights_sig)[0]
        norm_sig = np.sum( np.multiply(hist_sig, np.diff(bins) ) )
        staterrors_sig = np.sqrt(np.histogram(scores_sig, bins=bins, weights=np.square(weights_sig))[0])
        ax.stairs(hist_sig/norm_sig, edges=bins,
                  color='orange', label='Signal', linewidth=3)
        ax.stairs((hist_sig+staterrors_sig)/norm_sig, baseline=(hist_sig-staterrors_sig)/norm_sig,
                    color='orange', edges=bins, fill=True, alpha=0.3)
        # other plot settings
        ax.set_xlabel('Classifier output score', fontsize=12)
        ax.set_ylabel('Events (normalized)', fontsize=12)
        ax.set_title('Classifier output scores for sig and bkg', fontsize=12)
        ax.text(0.95, 0.8, 'AUC: {:.3f}'.format(auc), fontsize=12,
          ha='right', va='top', transform=ax.transAxes)
        if outputtag is not None:
            ax.text(0.95, 0.7, outputtag, fontsize=12,
            ha='right', va='top', transform=ax.transAxes)
        leg = ax.legend()
        figname = os.path.join(outputdir, 'scores.png')
        if outputtag is not None: figname = figname.replace('.png', f'_{outputtag}.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
        ax.set_yscale('log')
        figname = figname.replace('.png', f'_log.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')

    # calculate signal and background efficiency
    thresholds = np.linspace(np.amin(scores), np.amax(scores), num=100)
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
    if plot_roc and outputdir is not None:
        print('Making output score distribution...')
        fig, ax = plt.subplots()
        ax.plot(efficiency_bkg, efficiency_sig,
          color='dodgerblue', linewidth=3, label='ROC')
        ax.plot(efficiency_bkg, efficiency_bkg,
          color='darkblue', linewidth=3, linestyle='--', label='Baseline')
        ax.set_xlabel('Background pass-through', fontsize=12)
        ax.set_ylabel('Signal efficiency', fontsize=12)
        ax.text(0.95, 0.3, 'AUC: {:.3f}'.format(auc), fontsize=12,
          ha='right', va='bottom', transform=ax.transAxes)
        if outputtag is not None:
            ax.text(0.95, 0.2, outputtag, fontsize=12,
            ha='right', va='top', transform=ax.transAxes)
        leg = ax.legend()
        figname = os.path.join(outputdir, 'roc.png')
        if outputtag is not None: figname = figname.replace('.png', f'_{outputtag}.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')

    # close all figures
    plt.close()


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-y', '--score_branch', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('-s', '--signal_categories', default=[], nargs='+')
    parser.add_argument('-b', '--background_categories', default=[], nargs='+')
    parser.add_argument('--plot_score_dist', default=False, action='store_true')
    parser.add_argument('--plot_roc', default=False, action='store_true')
    args = parser.parse_args()

    # load events
    events = get_events_from_file(args.inputfile,
              treename = args.treename,
              signal_branches = args.signal_categories,
              background_branches = args.background_categories)

    # loop over signal and background categories
    for signal_category in args.signal_categories:
        for background_category in args.background_categories:
            print(f'Now running on signal branch {signal_category}'
                    + f' and background branch {background_category}...')

            # plot ROC
            plot_roc_from_events(events,
                outputdir = args.outputdir,
                score_branch = args.score_branch,
                signal_branch = signal_category,
                background_branch = background_category,
                plot_score_dist = args.plot_score_dist,
                plot_roc = args.plot_roc)
