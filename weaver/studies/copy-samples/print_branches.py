# Print branches in an ntuple


import os
import sys
import uproot
import argparse
import awkward as ak


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-t', '--treename', required=True)
    args = parser.parse_args()

    # read tree
    with uproot.open(f'{args.inputfile}:{args.treename}') as tree:

        # print branch info
        for bname, btype in tree.typenames().items():
            print(f'  {bname} ({btype})')
