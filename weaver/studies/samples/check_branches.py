# Print branches in an ntuple


import os
import sys
import uproot
import argparse
import numpy as np


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
    parser.add_argument('-t', '--treename', required=True)
    parser.add_argument('-b', '--branches', default=['auto'], nargs='+')
    args = parser.parse_args()

    # format branches to check
    if len(args.branches)==1 and args.branches[0]=='auto':
        # use hard-coded values
        args.branches = ([
          'passL1unprescaled_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65',
          'passTrigObjMatching_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65',
          'passTrig_HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65',
        ])

    # loop over input files
    results = np.zeros((len(args.inputfiles), len(args.branches)))
    for fileidx, inputfile in enumerate(args.inputfiles):

        # read tree
        with uproot.open(f'{inputfile}:{args.treename}') as tree:

            # find if branches are present
            for bidx, bname in enumerate(args.branches):
                if bname in tree.keys(): results[fileidx, bidx] = 1

    # number of present branches per file
    print('Number of branches per file:')
    for fileidx, inputfile in enumerate(args.inputfiles):
        npresent = int(np.sum(results[fileidx,:]))
        print(f'  - {inputfile}: {npresent}/{len(args.branches)}')
