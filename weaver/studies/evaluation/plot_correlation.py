import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from evaluationtools import get_scores_from_events
from evaluationtools import get_discos_from_events
from evaluationtools import get_events_from_file


def plot_correlation_from_file(inputfile, treename=None,
        category_branch=None, variable_branches=None, **kwargs):
    ### calculate and plot correlation directly from a file

    # format branch names to read
    category_branches = [category_branch] if category_branch is not None else []
    variable_branches = variable_branch if variable_branches is not None else []
    correlation_branches = category_branches + variable_branches

    # read events
    events = get_events_from_file(inputfile, treename=treename,
               correlation_branches=correlation_branches)

    # plot correlation from events
    return plot_correlation_from_events(events,
             category_branch = category_branch,
             variable_branches = variable_branches,
             **kwargs)


def plot_correlation_from_events(events,
                                  outputdir = None,
                                  score_branch = None,
                                  category_branch = None,
                                  variable_branches = None,
                                  calculate_disco = False,
                                  disco_npoints = 1000,
                                  disco_niterations = 5,
                                  plot_correlation = False,
                                  plot_correlation_slices = None):

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    if variable_branches is None: raise Exception('Must provide at least one variable branch.')
    if category_branch is not None and category_branch.lower() == "none": category_branch = None
    
    # get scores and mask for category
    # note: if category_branch is None, the returned labels are None,
    #       and no masking will be performed.
    # note: the same is true if category_branch is "none".
    (scores, labels) = get_scores_from_events(events, score_branch,
                         signal_branch=category_branch)
    if category_branch is not None: mask = (labels == 1)

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # calculate distance correlations
    if calculate_disco:
        dccoeffs = get_discos_from_events(events,
                     score_branch=score_branch, variable_branches=variable_branches,
                     npoints=disco_npoints, niterations=disco_niterations,
                     mask_branch=category_branch)

    # format label and tag for plots
    label = category_branch
    if category_branch is None: label = 'All events'
    else: label += ' events'
    tag = category_branch
    if category_branch is None: tag = 'all'

    # loop over variables
    for varname in variable_branches:

        # print disco
        if calculate_disco:
            msg = f'Distance correlation coefficient between scores and {varname}:'
            msg += ' {:.5f}'.format(dccoeffs[varname])
            print(msg)

        # make a plot of the correlation
        if plot_correlation and outputdir is not None:
            cvar = events[varname]
            cscores = scores
            if category_branch is not None:
                cvar = cvar[mask]
                cscores = cscores[mask]
            fig, ax = plt.subplots()
            ax.scatter(cvar, cscores,
                color='dodgerblue', label=label, alpha=0.3, s=1)
            ax.set_xlabel(varname, fontsize=12)
            ax.set_ylabel('Classifier output score', fontsize=12)
            ax.set_title(f'Correlation between {varname} and classifier output score', fontsize=12)
            txt = ax.text(0.95, 0.3, label, fontsize=12,
                    ha='right', va='top', transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='white', alpha=0.5))
            if calculate_disco:
                txt = ax.text(0.95, 0.2, 'DisCo: {:.3f}'.format(dccoeffs[varname]), fontsize=12,
                        ha='right', va='top', transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='white', alpha=0.5))
            figname = os.path.join(outputdir, f'correlation_{tag}_{varname}_scatter.png')
            fig.savefig(figname)
            print(f'Saved figure {figname}.')

        # same as above but in 2D histogram format instead of scatter plot
        if plot_correlation and outputdir is not None:
            cvar = events[varname]
            cscores = scores
            if category_branch is not None:
                cvar = cvar[mask]
                cscores = cscores[mask]
            score_bins = np.linspace(np.amin(cscores), np.amax(cscores), num=101)
            var_bins = np.linspace(0, 400, num=41) # ad hoc hardcoded...
            hist = np.histogram2d(cscores, cvar, bins=(score_bins, var_bins))[0]
            hist /= np.amax(hist)
            fig, ax = plt.subplots()
            im = ax.imshow(hist, cmap='plasma', interpolation='none', origin='lower', aspect='auto',
                        extent=(var_bins[0], var_bins[-1], score_bins[0], score_bins[-1]),
                        norm=mpl.colors.LogNorm(vmin=1e-2, vmax=1))
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(varname, fontsize=12)
            ax.set_ylabel('Classifier output score', fontsize=12)
            ax.set_title(f'Correlation between {varname} and classifier output score', fontsize=12)
            txt = ax.text(0.95, 0.3, label, fontsize=12,
                      ha='right', va='top', transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='white', alpha=0.5))
            if calculate_disco:
                txt = ax.text(0.95, 0.2, 'DisCo: {:.3f}'.format(dccoeffs[varname]), fontsize=12,
                          ha='right', va='top', transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='white', alpha=0.5))
            figname = os.path.join(outputdir, f'correlation_{tag}_{varname}_hist.png')
            fig.savefig(figname)
            print(f'Saved figure {figname}.')

        # plot correlation variables in slices of the score
        if plot_correlation_slices is not None and outputdir is not None:
            plot_correlation_slices = [float(el) for el in plot_correlation_slices]
        
            # loop over slices
            slices = []
            labels = []
            for idx in range(len(plot_correlation_slices)-1):
                minscore = plot_correlation_slices[idx]
                maxscore = plot_correlation_slices[idx+1]

                # get correlation variable in slice 
                slicemask = ((scores > minscore) & (scores < maxscore))
                if category_branch is not None: slicemask = ((slicemask) & (mask))
                cvar = events[varname][slicemask]
                slices.append(cvar)
                labels.append('{:.2f} < score < {:.2f}'.format(minscore, maxscore))

            # make a plot
            fig, ax = plt.subplots()
            #bins = np.histogram(events[varname], bins=20)[1]
            bins = np.linspace(0, 400, num=41) # ad hoc hardcoded...
            cmap = plt.get_cmap('cool', len(slices))
            for idx, (cslice, clabel) in enumerate(zip(slices, labels)):
                ax.hist(cslice, bins=bins, density=True,
                  color=cmap(idx), label=clabel,
                  histtype='step', linewidth=2)
            ax.set_xlabel(varname, fontsize=12)
            ax.set_ylabel('Events (normalized)', fontsize=12)
            ax.set_title(f'Correlation between {varname} and classifier output score', fontsize=12)
            txt = ax.text(0.95, 0.3, label, fontsize=12,
                    ha='right', va='top', transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='white', alpha=0.5))
            if calculate_disco:
                txt = ax.text(0.95, 0.2, 'DisCo: {:.3f}'.format(dccoeffs[varname]), fontsize=12,
                        ha='right', va='top', transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='white', alpha=0.5))
            leg = ax.legend(fontsize=12)
            for lh in leg.legend_handles:
                lh.set_alpha(1)
                lh._sizes = [30]
            fig.tight_layout()
            figname = os.path.join(outputdir, f'correlation_{tag}_{varname}_slices.png')
            fig.savefig(figname)
            print(f'Saved figure {figname}.')


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-y', '--score_branch', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('-c', '--categories', default=None, nargs='+')
    parser.add_argument('-v', '--variables', default=[], nargs='+')
    parser.add_argument('--calculate_disco', default=False, action='store_true')
    parser.add_argument('--plot_correlation', default=False, action='store_true')
    parser.add_argument('--plot_correlation_slices', default=None, nargs='+')
    args = parser.parse_args()

    # load events
    correlation_branches = args.variables[:]
    if args.categories is not None:
        for category in args.categories:
            if category.lower()=="none": continue
            correlation_branches.append(category)
    events = get_events_from_file(args.inputfile,
              treename = args.treename,
              correlation_branches = correlation_branches)

    # format correlation slices
    if args.plot_correlation_slices is not None:
        args.plot_correlation_slices = [float(el) for el in args.plot_correlation_slices]

    # format categories
    if args.categories is None: args.categories = [None]

    # loop over categories and variables
    for category in args.categories:
        print(f'Now running on category {category}...')

        # plot correlations
        plot_correlation_from_events(events,
                                  outputdir = args.outputdir,
                                  score_branch = args.score_branch,
                                  category_branch = category,
                                  variable_branches = args.variables,
                                  calculate_disco = args.calculate_disco,
                                  plot_correlation = args.plot_correlation,
                                  plot_correlation_slices = args.plot_correlation_slices)
