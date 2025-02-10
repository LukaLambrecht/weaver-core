import os
import sys
import uproot
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(weavercoredir)

from weaver.utils.disco import distance_correlation


def get_scores_from_events(events, score_branch=None,
        signal_branch=None, background_branch=None):
    ### get scores from an events array

    # format the scores
    if score_branch is None: raise Exception('Must provide a score branch.')
    scores = events[score_branch]

    # format the labels
    # note: if either the signal_branch or the background_branch is not specified,
    #       it is just taken to be the complement of the one that is specified.
    # note: if neither the signal branch nor the background_branch are specified,
    #       the labels are set to None.
    labels = -np.ones(len(scores))
    if signal_branch is not None:
        labels = np.where(events[signal_branch]==1, 1, labels)
        if background_branch is not None:
            labels = np.where(events[background_branch]==1, 0, labels)
        else: labels = np.where(labels==-1, 0, labels)
    elif background_branch is not None:
        labels = np.where(events[background_branch]==1, 0, 1)
    else: labels = None

    # mask the scores and labels
    # (i.e. remove entries that are neither signal nor background)
    if labels is not None:
        mask = (labels >= 0)
        scores = scores[mask]
        labels = labels[mask]

    # return the result
    return (scores, labels)


def get_discos_from_events(events, score_branch=None, variable_branches=None,
        npoints=1000, niterations=1, mask_branch=None):
    ### get distance correlation coefficient from events

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    if variable_branches is None: raise Exception('Must provide at least one variable branch.')

    # get scores
    # note: if mask_branch is None, the returned labels are None,
    #       and no masking will be performed.
    (scores, labels) = get_scores_from_events(events,
                         score_branch=score_branch,
                         signal_branch=mask_branch)
    if mask_branch is not None:
        mask = (labels == 1).astype(bool)

    # get requested variable branches from events
    variables = {b: events[b] for b in variable_branches}

    # mask scores and other branches if requested
    if mask_branch is not None:
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


def get_events_from_file(rootfile, treename=None,
        signal_branches=None, background_branches=None, correlation_branches=None):
    ### get scores and auxiliary variables from a root file

    # open input file
    fopen = rootfile
    if treename is not None: fopen += f':{treename}'
    events = uproot.open(fopen)

    # read branches as dict of arrays
    score_branches = [b for b in events.keys() if b.startswith('score_')]
    if signal_branches is None: signal_branches = []
    if background_branches is None: background_branches = []
    if correlation_branches is None: correlation_branches = []
    branches_to_read = score_branches + signal_branches + background_branches + correlation_branches
    branches_to_read = list(set(branches_to_read))
    events = events.arrays(branches_to_read, library='np')

    # return the events array
    return events
