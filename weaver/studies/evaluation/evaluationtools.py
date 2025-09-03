import os
import sys
import uproot
import numpy as np


def get_discos_from_events(events,
        score_branch=None, correlation_variables=None,
        npoints=1000, niterations=1, mask=None, mask_branch=None):
    ### get distance correlation coefficient from events

    # import distance correlation function
    thisdir = os.path.abspath(os.path.dirname(__file__))
    weaverdir = os.path.abspath(os.path.join(thisdir, '../../'))
    sys.path.append(weaverdir)
    from utils.disco import distance_correlation

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    if correlation_variables is None: raise Exception('Must provide at least one variable.')

    # make a mask
    if mask_branch is not None:
        if mask is None:
            mask = (events[mask_branch]).astype(bool)
        else:
            mask = ((mask) & (events[mask_branch]).astype(bool))

    # get scores
    scores = events[score_branch]

    # get requested variable branches from events
    variables = {varname: events[var['branch']] 
                  for varname, var in correlation_variables.items()}

    # mask scores and other branches if requested
    if mask is not None:
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
    for varname in variables.keys():
        dccoeffs[varname] = (np.mean(dccoeffs[varname]), np.std(dccoeffs[varname]))

    return dccoeffs


def get_events_from_file(rootfile, treename=None, branches=None):
    ### get scores and auxiliary variables from a root file

    # open input file
    fopen = rootfile
    if treename is not None: fopen += f':{treename}'
    events = uproot.open(fopen)

    # set branches to read
    events = events.arrays(branches, library='np')

    # return the events array
    return events
