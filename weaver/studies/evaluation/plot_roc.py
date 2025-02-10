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
        signal_branch=None, background_branch=None, **kwargs):
    ### calculate and plot ROC directly from an input file

    # format the signal and background branch names
    signal_branches = [signal_branch] if signal_branch is not None else []
    background_branches = [background_branch] if background_branch is not None else []

    # read events
    events = get_events_from_file(inputfile, treename=treename,
               signal_branches=signal_branches, background_branches=background_branches)

    # plot ROC from events
    return plot_roc_from_events(events, signal_branch=signal_branch,
            background_branch=background_branch, **kwargs)


def plot_roc_from_events(events,
        score_branch=None, signal_branch=None, background_branch=None, **kwargs):
    ### calculate and plot ROC from an events array

    # format the score, signal, and background branch names
    if score_branch is None:
        raise Exception('A score branch must be specified.')
    if signal_branch is None and background_branch is None:
        raise Exception('A signal branch or a background branch (or both) must be specified.')

    # get scores and labels
    (scores, labels) = get_scores_from_events(events,
                         score_branch=score_branch,
                         signal_branch=signal_branch,
                         background_branch=background_branch)

    # plot ROC from scores
    outputtag = f'{signal_branch}_vs_{background_branch}'
    return plot_roc_from_scores(scores, labels, outputtag=outputtag, **kwargs)


def plot_roc_from_scores(scores, labels,
             outputdir=None,
             outputtag=None,
             plot_score_dist=False,
             plot_roc=False):

    # separate scores into signal and background scores
    scores_sig = scores[labels==1]
    scores_bkg = scores[labels==0]

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # calculate AUC
    print('Calculating AUC (ROC)...')
    auc = roc_auc_score(labels, scores)
    print('AUC (ROC) on testing set: {:.3f}'.format(auc))

    # make a plot of the score distribution
    if plot_score_dist and outputdir is not None:
        print('Making output score distribution...')
        fig, ax = plt.subplots()
        bins = np.linspace(np.amin(scores_bkg), np.amax(scores_sig), num=50)
        ax.hist(scores_bkg, bins=bins, color='dodgerblue', histtype='step', linewidth=3, label='Background')
        ax.hist(scores_sig, bins=bins, color='orange', histtype='step', linewidth=3, label='Signal')
        ax.set_xlabel('Classifier output score', fontsize=12)
        ax.set_ylabel('Number of events', fontsize=12)
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

    # calculate signal and background efficiency
    thresholds = np.linspace(np.amin(scores), np.amax(scores), num=100)
    efficiency_sig = np.zeros(len(thresholds))
    efficiency_bkg = np.zeros(len(thresholds))
    for idx, threshold in enumerate(thresholds):
        eff_s = np.sum(scores_sig > threshold)
        efficiency_sig[idx] = eff_s
        eff_b = np.sum(scores_bkg > threshold)
        efficiency_bkg[idx] = eff_b
    efficiency_sig /= len(scores_sig)
    efficiency_bkg /= len(scores_bkg)

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
