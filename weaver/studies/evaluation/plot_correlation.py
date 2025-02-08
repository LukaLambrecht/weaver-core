import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from evaluationtools import get_scores_from_events
from evaluationtools import get_discos_from_events
from evaluationtools import get_events_from_file


def plot_correlation_from_file(inputfile, signal_mask, outputdir,
        treename=None, correlations=[], **kwargs):
    events = get_events_from_file(inputfile, signal_mask,
               treename=treename, correlations=correlations)
    return plot_correlations_from_events(events, signal_mask, outputdir,
            correlations=correlations, **kwargs)

def plot_correlation_from_events(events, signal_mask, outputdir,
                                  correlations = [],
                                  calculate_disco = False,
                                  disco_npoints = 1000,
                                  disco_niterations = 5,
                                  plot_correlation = False,
                                  plot_correlation_slices = None):

    # get scores and set mask for bkg only
    (scores, labels) = get_scores_from_events(events, signal_mask)
    mask_bkg = (labels == 0)

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # do nothing if no correlation variables are provided
    if len(correlations)==0: return

    # calculate distance correlations
    if calculate_disco:
        dccoeffs = get_discos_from_events(events, scores, correlations,
                     npoints=disco_npoints, niterations=disco_niterations,
                     mask=mask_bkg)

    # loop over variables
    for cvarname in correlations:

        # print disco
        if calculate_disco:
            msg = f'Distance correlation coefficient between scores and {cvarname}:'
            msg += ' {:.5f}'.format(dccoeffs[cvarname])
            print(msg)

        # make a plot of the correlation
        if plot_correlation:
            cvar = events[cvarname][mask_bkg]
            cscores = scores[mask_bkg]
            fig, ax = plt.subplots()
            label = 'Bkg'
            if calculate_disco: label += ' (disco: {:.3f})'.format(dccoeffs[cvarname])
            ax.scatter(cvar, cscores,
                color='dodgerblue', label=label, alpha=0.5, s=1)
            ax.set_xlabel('Mass (GeV)', fontsize=12)
            ax.set_ylabel('Classifier output score', fontsize=12)
            ax.set_title(f'Correlation between {cvarname} and classifier output score', fontsize=12)
            leg = ax.legend(fontsize=12)
            for lh in leg.legend_handles:
                lh.set_alpha(1)
                lh._sizes = [30]
            fig.savefig(os.path.join(outputdir, f'correlation_{cvarname}.png'))

        # plot correlation variables in slices of the score
        if plot_correlation_slices is not None:
            plot_correlation_slices = [float(el) for el in plot_correlation_slices]
        
            # loop over slices
            slices = []
            labels = []
            for idx in range(len(plot_correlation_slices)-1):
                minscore = plot_correlation_slices[idx]
                maxscore = plot_correlation_slices[idx+1]

                # get correlation variable in slice 
                mask = ((mask_bkg) & (scores > minscore) & (scores < maxscore))
                cvar = events[cvarname][mask]
                slices.append(cvar)
                labels.append('{:.2f} < score < {:.2f}'.format(minscore, maxscore))

            # make a plot
            fig, ax = plt.subplots()
            #bins = np.histogram(events[cvarname], bins=20)[1]
            bins = np.linspace(0, 400, num=41) # ad hoc
            cmap = plt.get_cmap('cool', len(slices))
            for idx, (cslice, label) in enumerate(zip(slices, labels)):
                ax.hist(cslice, bins=bins, density=True,
                  color=cmap(idx), label=label,
                  histtype='step', linewidth=2)
            ax.set_xlabel('Mass (GeV)', fontsize=12)
            ax.set_ylabel('Events (normalized)', fontsize=12)
            ax.set_title(f'Correlation between {cvarname} and classifier output score', fontsize=12)
            leg = ax.legend(fontsize=12)
            for lh in leg.legend_handles:
                lh.set_alpha(1)
                lh._sizes = [30]
            fig.tight_layout()
            fig.savefig(os.path.join(outputdir, f'correlation_slices_{cvarname}.png'))


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-s', '--signal_mask', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('-c', '--correlations', default=[], nargs='+')
    parser.add_argument('--calculate_disco', default=False, action='store_true')
    parser.add_argument('--plot_correlation', default=False, action='store_true')
    parser.add_argument('--plot_correlation_slices', default=None, nargs='+')
    args = parser.parse_args()

    # call main function
    plot_correlations_from_file(args.inputfile, args.signal_mask, args.outputdir,
             treename = args.treename, correlations = args.correlations,
             calculate_disco = args.calculate_disco,
             plot_correlation = args.plot_correlation,
             plot_correlation_slices = args.plot_correlation_slices)
