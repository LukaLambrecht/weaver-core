# Do model evaluation specifically for synthetic data vs data models

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
evaldir = os.path.abspath(os.path.join(thisdir, '../evaluation'))
sys.path.append(evaldir)

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
    treename = 'Events'
    score_branch = 'score_isSignal'
    signal_categories = {
        'syndata': {
            'branch': 'isSynData',
            'color': 'blue',
            'label': r'Syn. data'
        }
    }
    background_categories = {
        'data': {
            'branch': 'isData',
            'color': 'green',
            'label': 'Data'
        }
    }
    all_categories = {**signal_categories, **background_categories}
    score_secondary_variable = None
    phase_space_split = {
      'all': {},
      '3T': {'dHH_NbtagT': [2.5, 4.5]},
    }

    # find all branches to read
    branches_to_read = [score_branch]
    branches_to_read += [cat['branch'] for cat in all_categories.values()]
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
                xsecweighting = False,
                outputdir = outputdir,
                score_branch = score_branch)
            plt.close()
            plot_roc(this_events,
                signal_categories, background_categories,
                xsecweighting = False,
                outputdir = outputdir,
                score_branch = score_branch)
            plt.close()
