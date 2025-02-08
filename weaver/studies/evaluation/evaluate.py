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
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-s', '--signal_mask', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('-c', '--correlations', default=[], nargs='+')
    parser.add_argument('--plot_score_dist', default=False, action='store_true')
    parser.add_argument('--plot_roc', default=False, action='store_true')
    parser.add_argument('--calculate_disco', default=False, action='store_true')
    parser.add_argument('--plot_correlation', default=False, action='store_true')
    parser.add_argument('--plot_correlation_slices', default=None, nargs='+')
    args = parser.parse_args()

    # load events
    events = get_events_from_file(args.inputfile, args.signal_mask,
              treename=args.treename, correlations=args.correlations)

    # plot roc
    plot_roc_from_events(events, args.signal_mask, args.outputdir,
             plot_score_dist = args.plot_score_dist,
             plot_roc = args.plot_roc)

    # plot correlations
    if args.plot_correlation_slices is not None:
        args.plot_correlation_slices = [float(el) for el in args.plot_correlation_slices]
    plot_correlation_from_events(events, args.signal_mask, args.outputdir,
                                  correlations = args.correlations,
                                  calculate_disco = args.calculate_disco,
                                  plot_correlation = args.plot_correlation,
                                  plot_correlation_slices = args.plot_correlation_slices)
