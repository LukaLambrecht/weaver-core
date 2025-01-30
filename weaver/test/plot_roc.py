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

    # open input file
    fopen = args.inputfile
    if args.treename is not None: fopen += f':{args.treename}'
    events = uproot.open(fopen)

    # read branches as dict of arrays
    score_branches = [b for b in events.keys() if b.startswith('score_')]
    mask_branches = [args.signal_mask] #+ [args.background_mask]
    branches_to_read = score_branches + mask_branches + args.correlations
    events = events.arrays(branches_to_read, library='np')

    # format the scores
    scores = events['score_'+args.signal_mask]
    labels = np.where(events[args.signal_mask]==1, 1, 0)
    scores_sig = scores[labels==1]
    scores_bkg = scores[labels==0]

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
        for cvarname in args.correlations:

            # get variable
            cvar = events[cvarname]
            cvar_sig = cvar[labels==1]
            cvar_bkg = cvar[labels==0]

            # calculate correlation coefficient
            limit = 1000 # cannot calculate the distance correlation on the full set, too much memory
            randinds = np.random.choice(np.arange(len(cvar_bkg)), size=limit, replace=False)
            dccoeff = distance_correlation(cvar_bkg[randinds], scores_bkg[randinds])
            print('Distance correlation coefficient: {:.5f}'.format(dccoeff))

            # make a plot of the correlation
            if args.plot_correlation:
                fig, ax = plt.subplots()
                label = 'Bkg (disco: {:.3f})'.format(dccoeff)
                ax.scatter(cvar_bkg, scores_bkg,
                  color='dodgerblue', label=label, alpha=0.5, s=1)
                ax.set_xlabel('Mass (GeV)', fontsize=12)
                ax.set_ylabel('Classifier output score', fontsize=12)
                ax.set_title(f'Correlation between {cvarname} and classifier output score', fontsize=12)
                leg = ax.legend(fontsize=12)
                for lh in leg.legend_handles:
                    lh.set_alpha(1)
                    lh._sizes = [30]
                fig.savefig(os.path.join(args.outputdir, f'correlation_{cvarname}.png'))
