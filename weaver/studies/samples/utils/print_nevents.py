# Print number of events in ntuples


import os
import sys
import uproot
import argparse
import awkward as ak


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
    parser.add_argument('-t', '--treename', required=True)
    args = parser.parse_args()

    # loop over input files
    nevents = {}
    for inputfile in args.inputfiles:

        # read tree
        with uproot.open(f'{inputfile}:{args.treename}') as tree:

            # get number of events
            nevents[inputfile] = tree.num_entries

    # do printouts
    keys = sorted(list(nevents.keys()))
    print('Number of events:')
    for key in keys:
        print(f'  - {key}: {nevents[key]}')
    print(f'Total: {sum(list(nevents.values()))}')
