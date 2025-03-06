import os
import sys
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from evaluationtools import get_events_from_file
from plot_roc import plot_roc_from_events
from plot_correlation import plot_correlation_from_events


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
    parser.add_argument('-y', '--score_branch', required=True)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('-s', '--signal_categories', default=[], nargs='+')
    parser.add_argument('-b', '--background_categories', default=[], nargs='+')
    parser.add_argument('-c', '--correlation_categories', default=[], nargs='+')
    parser.add_argument('-v', '--correlation_variables', default=[], nargs='+')
    parser.add_argument('--xsecweighting', default=False, action='store_true')
    parser.add_argument('--plot_score_dist', default=False, action='store_true')
    parser.add_argument('--plot_roc', default=False, action='store_true')
    parser.add_argument('--calculate_disco', default=False, action='store_true')
    parser.add_argument('--plot_correlation', default=False, action='store_true')
    parser.add_argument('--plot_correlation_slices', default=None, nargs='+')
    args = parser.parse_args()

    # loop over input files
    for inputfile in args.inputfiles:

        # set output directory
        outputdir = os.path.splitext(inputfile)[0] + '_plots'

        # load events
        correlation_branches = args.correlation_categories + args.correlation_variables
        weight_branches = ['genWeight', 'xsecWeight'] if args.xsecweighting else None
        events = get_events_from_file(inputfile,
                  treename = args.treename,
                  signal_branches = args.signal_categories,
                  background_branches = args.background_categories,
                  correlation_branches = correlation_branches,
                  weight_branches = weight_branches)

        # loop over signal and background categories
        # for ROC plotting
        for signal_category in args.signal_categories:
            for background_category in args.background_categories:

                # plot ROC
                plot_roc_from_events(events,
                    xsecweighting = args.xsecweighting,
                    outputdir = outputdir,
                    score_branch = args.score_branch,
                    signal_branch = signal_category,
                    background_branch = background_category,
                    plot_score_dist = args.plot_score_dist,
                    plot_roc = args.plot_roc)

        # format correlation slices
        if args.plot_correlation_slices is not None:
            args.plot_correlation_slices = [float(el) for el in args.plot_correlation_slices]

        # loop over categories for correlation plotting
        for category in args.correlation_categories:

            # plot correlations
            plot_correlation_from_events(events,
                                  xsecweighting = args.xsecweighting,
                                  outputdir = outputdir,
                                  score_branch = args.score_branch,
                                  category_branch = category,
                                  variable_branches = args.correlation_variables,
                                  calculate_disco = args.calculate_disco,
                                  plot_correlation = args.plot_correlation,
                                  plot_correlation_slices = args.plot_correlation_slices)
