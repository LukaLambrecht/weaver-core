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


def plot_roc_from_file(inputfile, signal_mask, outputdir, treename=None, **kwargs):
    events = get_events_from_file(inputfile, signal_mask, treename=treename)
    return plot_roc_from_events(events, signal_mask, outputdir, **kwargs)

def plot_roc_from_events(events, signal_mask, outputdir, **kwargs):
    (scores, labels) = get_scores_from_events(events, signal_mask)
    return plot_roc_from_scores(scores, labels, outputdir, **kwargs)

def plot_roc_from_scores(scores, labels, outputdir,
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
    if plot_score_dist:
        print('Making output score distribution...')
        fig, ax = plt.subplots()
        bins = np.linspace(np.amin(scores_bkg), np.amax(scores_sig), num=50)
        ax.hist(scores_bkg, bins=bins, color='dodgerblue', histtype='step', linewidth=3, label='Background')
        ax.hist(scores_sig, bins=bins, color='orange', histtype='step', linewidth=3, label='Signal')
        ax.set_xlabel('Classifier output score', fontsize=12)
        ax.set_ylabel('Number of events', fontsize=12)
        ax.set_title('Classifier output scores for sig and bkg', fontsize=12)
        ax.text(0.95, 0.8, 'AUC: {:.3f}'.format(auc), fontsize=15,
          ha='right', va='top', transform=ax.transAxes)
        leg = ax.legend()
        fig.savefig(os.path.join(outputdir,'scores.png'))

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
    if plot_roc:
        print('Making output score distribution...')
        fig, ax = plt.subplots()
        ax.plot(efficiency_bkg, efficiency_sig,
          color='dodgerblue', linewidth=3, label='ROC')
        ax.plot(efficiency_bkg, efficiency_bkg,
          color='darkblue', linewidth=3, linestyle='--', label='Baseline')
        ax.set_xlabel('Background pass-through', fontsize=12)
        ax.set_ylabel('Signal efficiency', fontsize=12)
        ax.text(0.95, 0.2, 'AUC: {:.3f}'.format(auc), fontsize=15,
          ha='right', va='bottom', transform=ax.transAxes)
        leg = ax.legend()
        fig.savefig(os.path.join(outputdir,'roc.png'))


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-s', '--signal_mask', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('--plot_score_dist', default=False, action='store_true')
    parser.add_argument('--plot_roc', default=False, action='store_true')
    args = parser.parse_args()

    # call main function
    plot_roc_from_file(args.inputfile, args.signal_mask, args.outputdir,
             treename = args.treename,
             plot_score_dist = args.plot_score_dist,
             plot_roc = args.plot_roc)
