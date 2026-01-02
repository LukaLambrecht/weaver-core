# Simple utility script to print the number of events in a root file


import os
import sys
import uproot


if __name__=='__main__':

    # read input files
    inputfiles = sys.argv[1:]
    #treename = 'tree' # for input ntuples
    treename = 'Events' # for output files

    # get number of events per file
    nevents = {}
    for inputfile in inputfiles:
        f = uproot.open(inputfile+':'+treename)
        nevents[inputfile] = f.num_entries

    # print results
    print('Found following number of events:')
    for key, val in nevents.items():
        print(f'  - {key}: {val}')
    print(f'  --> total: {sum(list(nevents.values()))}')
