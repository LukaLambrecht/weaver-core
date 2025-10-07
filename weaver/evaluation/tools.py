import os
import sys
import uproot
import numpy as np


def read_file(rootfile, treename=None, branches=None):
    ### get scores and auxiliary variables from a root file

    # open input file
    fopen = rootfile
    if treename is not None: fopen += f':{treename}'
    events = uproot.open(fopen)

    # set branches to read
    events = events.arrays(branches, library='np')

    # return the events array
    return events
