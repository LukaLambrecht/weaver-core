# Do model evaluation specifically for HHto4b vs QCD/TT models

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from evaluationtools import get_events_from_file
from plot_roc import plot_scores, plot_roc
from plot_roc_multi import plot_scores_multi, plot_roc_multi
from plot_correlation import plot_correlation
from plot_correlation_multi import plot_correlation_multi


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
    args = parser.parse_args()

    # hard coded settings
    score_branch = 'score_isSignal'
    treename = 'Events'
    xsecweighting = True
    signal_categories = {
        'HH': {
            'branch': 'isSignal',
            'color': 'red',
            'label': r'HH $\rightarrow$ 4b'
        }
    }
    background_categories = {
        'QCD': {
            'branch': 'isQCD',
            'color': 'blue',
            'label': 'QCD'
        },
        'TT': {
            'branch': 'isTT',
            'color': 'green',
            'label': r't$\bar{t}$'
        }
    }
    all_categories = {**signal_categories, **background_categories}
    correlation_variables = {
        'mH1': {
            'branch': 'dHH_H1_mass',
            'label': '$m(H_{1})$ [GeV]',
            'bins': np.linspace(50, 250, num=26)
        },
        'mH2': {
            'branch': 'dHH_H2_mass',
            'label': '$m(H_{2})$ [GeV]',
            'bins': np.linspace(50, 250, num=26)
        },
        #'mHH': {
        #    'branch': 'dHH_HH_mass',
        #    'label': '$m(HH)$ [GeV]',
        #    'bins': np.linspace(350, 1000, num=26)
        #},
        'mHavg': {
            'branch': 'hh_average_mass',
            'label': '$m(H_{avg}) [GeV]$',
            'bins': np.linspace(50, 250, num=26)
        }
    }
    correlation_slices = [0., 0.5, 0.75, 1]
    phase_space_split = {
      'all': {},
      '3M': {'dHH_NbtagM': [2.5, 3.5]},
      '4M': {'dHH_NbtagM': [3.5, 4.5]}
    }

    # find all branches to read
    branches_to_read = (
        [score_branch]
        + [cat['branch'] for cat in signal_categories.values()]
        + [cat['branch'] for cat in background_categories.values()]
        + [v['branch'] for v in correlation_variables.values()]
    )
    if xsecweighting:
        branches_to_read += ['lumiwgt', 'genWeight', 'xsecWeight']
    for region_cuts in phase_space_split.values():
        for variable in region_cuts.keys(): branches_to_read.append(variable)

    # loop over input files
    for inputfile in args.inputfiles:
        print(f'Running on input file {inputfile}...')

        # load events
        events = get_events_from_file(
                   inputfile,
                   treename = treename,
                   branches = branches_to_read)
        keys = list(events.keys())
        nevents = len(events[list(events.keys())[0]])
        print('Read events file with following properties:')
        print(f'  - Keys: {keys}.')
        print(f'  - Number of events: {nevents}.')

        # loop over phase space splits
        for region_name, region_cuts in phase_space_split.items():

            # define output directory
            outputdir = inputfile.replace('.root', '_plots')
            outputdir = os.path.join(outputdir, region_name)

            # make a mask for this region
            mask = np.ones(nevents).astype(bool)
            for variable, (vmin, vmax) in region_cuts.items():
                mask = ((mask) & (events[variable]>vmin).astype(bool) & (events[variable]<vmax).astype(bool))
            print(f'  Phase space region {region_name}: selected {np.sum(mask)} / {nevents} events.')
            this_events = {key: val[mask] for key, val in events.items()}

            # plot the score distributions and ROC curves
            print(f'    Plotting ROC...')
            plot_scores(this_events,
                signal_categories, background_categories,
                xsecweighting = xsecweighting,
                outputdir = outputdir,
                score_branch = score_branch)
            plt.close()
            plot_roc(this_events,
                signal_categories, background_categories,
                xsecweighting = xsecweighting,
                outputdir = outputdir,
                score_branch = score_branch)
            plt.close()

            # plot correlations
            print(f'    Plotting correlations...')
            plot_correlation(this_events,
                all_categories,
                xsecweighting = xsecweighting,
                outputdir = outputdir,
                score_branch = score_branch,
                variables = correlation_variables,
                slices = correlation_slices)
            plt.close()

            # plot score distribution and ROC for multiple categories together
            plot_scores_multi(
                this_events,
                all_categories,
                xsecweighting = xsecweighting,
                outputdir = outputdir,
                score_branch = score_branch)
            plot_roc_multi(
                this_events,
                signal_categories,
                background_categories,
                xsecweighting = xsecweighting,
                outputdir = outputdir,
                score_branch = score_branch)

            # plot correlation for multiple categories together
            plot_correlation_multi(
                this_events,
                all_categories,
                xsecweighting = xsecweighting,
                outputdir = outputdir,
                score_branch = score_branch,
                score_bins = correlation_slices,
                variables = correlation_variables)
            plt.close()
