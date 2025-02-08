import os
import sys
import uproot
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(weavercoredir)

from weaver.utils.disco import distance_correlation


def get_scores_from_events(events, signal_mask):
    ### get scores from an events array

    # format the scores
    scores = events['score_'+signal_mask]
    labels = np.where(events[signal_mask]==1, 1, 0)

    # return the result
    return (scores, labels)


def get_discos_from_events(events, scores, correlations, npoints=1000, niterations=1, mask=None):
    ### get distance correlation coefficients from events

    # get requested branches from events
    variables = {c: events[c] for c in correlations}

    # mask scores and other branches if requested
    if mask is not None:
        mask = mask.astype(bool)
        scores = scores[mask]
        for varname, values in variables.items():
            variables[varname] = values[mask]

    # initialize indices for random selection of points
    # (to avoid excessive memory usage)
    randinds = [None]
    if npoints > 0 and len(scores) <= npoints:
        msg = 'WARNING in get_discos_from_events:'
        msg += f' found setting npoints = {npoints},'
        msg += f' but input scores has only length {len(scores)};'
        msg += f' will use npoints = {len(scores)} instead.'
        print(msg)
        pass
    elif npoints > 0 and len(scores) > npoints:
        replace = False
        totalnpoints = niterations * npoints
        if len(scores) <= totalnpoints:
            msg = 'WARNING in get_discos_from_events:'
            msg += f' found settings npoints = {npoints} and niterations = {niterations}'
            msg += f' but input scores has only length {len(scores)};'
            msg += f' will set replace = True in random index sampling.'
            print(msg)
            replace = True
        randinds = np.random.choice(np.arange(len(scores)), size=totalnpoints, replace=replace)
        randinds = np.split(randinds, niterations)

    # loop over iterations
    # (results will be averaged)
    dccoeffs = {}
    for varname in variables.keys(): dccoeffs[varname] = np.zeros(niterations)
    for iteridx, this_ids in enumerate(randinds):

        # select scores
        if this_ids is not None: this_scores = scores[this_ids]

        # loop over correlation variables
        for varname, values in variables.items():

            # select variable
            if this_ids is not None: this_values = values[this_ids]

            # calculate distance correlation
            dccoeff = distance_correlation(this_values, this_scores)
            dccoeffs[varname][iteridx] = dccoeff

    # do averaging over iterations
    for varname in variables.keys(): dccoeffs[varname] = np.mean(dccoeffs[varname])

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
