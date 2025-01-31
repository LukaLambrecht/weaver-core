import os
import sys
import uproot
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from sklearn.metrics import roc_auc_score

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

from weaver.utils.disco import distance_correlation


def get_scores_from_events(events, signal_mask):
    ### get scores from an events array

    # format the scores
    scores = events['score_'+signal_mask]
    labels = np.where(events[signal_mask]==1, 1, 0)
    scores_sig = scores[labels==1]
    scores_bkg = scores[labels==0]

    # return the result
    return (scores, labels, scores_sig, scores_bkg)


def get_discos_from_events(events, scores, correlations, npoints=1000, mask=None):
    ### get distance correlation coefficients from events

    # mask scores if requested
    if mask is not None:
        mask = mask.astype(bool)
        scores = scores[mask]

    # initialize indices for random selection of points
    # (to avoid excessive memory usage)
    randinds = None
    if npoints > 0 and len(scores) > npoints:
        randinds = np.random.choice(np.arange(len(scores)), size=npoints, replace=False)
        scores = scores[randinds]

    # loop over correlation variables
    dccoeffs = {}
    for varname in correlations:

        # get variable
        var = events[varname]
        if mask is not None: var = var[mask]
        if randinds is not None: var = var[randinds]

        # calculate distance correlation
        dccoeff = distance_correlation(var, scores)
        dccoeffs[varname] = dccoeff

    return dccoeffs


def get_events_from_file(rootfile, signal_mask, correlations=None, treename=None):
    ### get scores and auxiliary variables from a root file

    # open input file
    fopen = rootfile
    if treename is not None: fopen += f':{treename}'
    events = uproot.open(fopen)

    # read branches as dict of arrays
    score_branches = [b for b in events.keys() if b.startswith('score_')]
    mask_branches = [signal_mask]
    correlation_branches = correlations if correlations is not None else []
    branches_to_read = score_branches + mask_branches + correlation_branches
    events = events.arrays(branches_to_read, library='np')

    # return the events array
    return events


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-s', '--signal_mask', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('-c', '--correlations', default=[], nargs='+')
    parser.add_argument('--plot_score_dist', default=False, action='store_true')
    parser.add_argument('--plot_roc', default=False, action='store_true')
    parser.add_argument('--plot_correlation', default=False, action='store_true')
    args = parser.parse_args()

    # read events and scores from input file
    events = get_events_from_file(args.inputfile, args.signal_mask,
               correlations=args.correlations, treename=args.treename)
    (scores, labels, scores_sig, scores_bkg) = get_scores_from_events(events, args.signal_mask)

    # make output directory
    if args.outputdir is not None:
        if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

    # calculate AUC
    print('Calculating AUC (ROC)...')
    auc = roc_auc_score(labels, scores)
    print('AUC (ROC) on testing set: {:.3f}'.format(auc))

    # make a plot of the score distribution
    if args.plot_score_dist:
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
        fig.savefig(os.path.join(args.outputdir,'scores.png'))

    # calculate signal and background efficiency
    #thresholds = np.sort(scores)
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
    if args.plot_roc:
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
        fig.savefig(os.path.join(args.outputdir,'roc.png'))

    # calculate and plot correlations
    if len(args.correlations)>0:

        # calculate distance correlations
        mask = (labels == 0) # only for background events
        dccoeffs = get_discos_from_events(events, scores, args.correlations, npoints=1000, mask=mask)

        # loop over variables
        for cvarname in args.correlations:
            msg = f'Distance correlation coefficient between scores and {cvarname}:'
            msg += ' {:.5f}'.format(dccoeffs[cvarname])
            print(msg)

            # make a plot of the correlation
            if args.plot_correlation:
                cvar = events[cvarname][mask]
                cscores = scores[mask]
                fig, ax = plt.subplots()
                label = 'Bkg (disco: {:.3f})'.format(dccoeffs[cvarname])
                ax.scatter(cvar, cscores,
                  color='dodgerblue', label=label, alpha=0.5, s=1)
                ax.set_xlabel('Mass (GeV)', fontsize=12)
                ax.set_ylabel('Classifier output score', fontsize=12)
                ax.set_title(f'Correlation between {cvarname} and classifier output score', fontsize=12)
                leg = ax.legend(fontsize=12)
                for lh in leg.legend_handles:
                    lh.set_alpha(1)
                    lh._sizes = [30]
                fig.savefig(os.path.join(args.outputdir, f'correlation_{cvarname}.png'))
