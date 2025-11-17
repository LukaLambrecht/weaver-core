import os
import sys
import uproot
import argparse
import numpy as np
import awkward as ak
from fnmatch import fnmatch


def read_file(filename, treename=None, branches=None):
    '''
    Read an input file
    '''
    if treename is not None: filename += f':{treename}'
    with uproot.open(filename) as f:
        valid_branches = None
        if branches is not None:
            valid_branches = branches[:]
            for branch in branches:
                if not branch in f.keys():
                    valid_branches.remove(branch)
                    msg = 'WARNING in tools.samplelisttools.read_sampledict:'
                    msg += f' branch {branch} not found in {filename};'
                    msg += ' will skip reading this branch.'
                    print(msg)
        tree = f.arrays(valid_branches, library='ak')
    return tree


def make_writable_tree(tree, records=None, counters_to_remove=None):
    # helper function for write_tree

    # convert tree to format suitable for writing
    # (see here: https://github.com/scikit-hep/uproot5/discussions/903)
    writebranches = dict(zip(ak.fields(tree), ak.unzip(tree)))
    if records is not None:
        for recordname in records:
            tag = recordname + '_'
            recordbranches = {k: v for k, v in writebranches.items() if k.startswith(tag)}
            if len(recordbranches)==0: continue
            writebranches = {k: v for k, v in writebranches.items() if not k.startswith(tag)}
            record = ak.zip({name[len(tag):]: v for name, v in recordbranches.items()})
            writebranches[recordname] = record

    # need to remove counter branches as they are added automatically,
    # otherwise gives strange dtype errors...
    if counters_to_remove is not None:
        new_writebranches = {}
        for k, v in writebranches.items():
            keep = True
            for c in counters_to_remove:
                if fnmatch(k, c): keep = False
            if not keep: continue
            new_writebranches[k] = v
        writebranches = new_writebranches

    return writebranches


def write_tree(tree, rootfile, treename='tree', **kwargs):
    '''
    Write a tree to a ROOT file
    Input arguments:
      - tree: tree to write, in the format of an events dict
        of the form {'<variable name>': awkward array}.
      - rootfile: name of the file to write.
    '''

    # convert tree to writable format
    writebranches = make_writable_tree(tree, **kwargs)

    # write to file
    outputdir = os.path.dirname(rootfile)
    if len(outputdir)>0:
        if not os.path.exists(outputdir): os.makedirs(outputdir)
    with uproot.recreate(rootfile) as f:
        f[treename] = writebranches


def upsample_naive(jets, factor=1, std=0.1, copy_branches=None):
    '''
    Do naive upsampling of a collection of jets
    '''
    new_jets = {}
    rng = np.random.default_rng()
    for branch in jets.fields:
        arr_orig = jets[branch]
        ndim = arr_orig.layout.minmax_depth[1]

        # flatten
        if ndim>1:
            num_orig = ak.num(arr_orig).to_numpy()
            arr_orig = ak.flatten(arr_orig, axis=None).to_numpy()
            num_upsampled = np.repeat(num_orig, factor)
            
        # upsample
        arr_upsampled = np.repeat(arr_orig, factor)
        if branch in copy_branches: pass
        else:
            shifts = 1 + rng.normal(size=len(arr_upsampled))*std
            arr_upsampled = np.multiply(shifts, arr_upsampled)

        # unflatten
        if ndim>1: arr_upsampled = ak.unflatten(arr_upsampled, num_upsampled)

        # add to dict
        new_jets[branch] = arr_upsampled
    new_jets = ak.Array(new_jets)
    return new_jets


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-o', '--outputfile', required=True)
    parser.add_argument('-f', '--upsample_factor', default=1, type=int)
    parser.add_argument('-s', '--std', default=0.1, type=float)
    args = parser.parse_args()

    # set branches to read
    # (hard-coded for now, maybe later use json file)
    branches_to_read = [
      'recojet_isB',
      'recojet_isC',
      'recojet_isUDSG',
      'recojet_pt',
      'recojet_eta',
      'recojet_theta',
      'recojet_phi',
      'recojet_e',
      'recojet_mass',
      'nconst',
      'pfcand_px',
      'pfcand_py',
      'pfcand_pz',
      'pfcand_pt',
      'pfcand_e',
      'pfcand_ptrel_log',
      'pfcand_erel_log',
      'pfcand_charge',
      'pfcand_dxy',
      'pfcand_dz',
      'pfcand_btagSip2dVal',
      'pfcand_btagSip2dSig',
      'pfcand_btagSip3dVal',
      'pfcand_btagSip3dSig',
      'pfcand_btagJetDistVal',
      'pfcand_btagJetDistSig',
      'pfcand_isChargedHad',
      'pfcand_isNeutralHad',
      'pfcand_isGamma',
      'pfcand_isEl',
      'pfcand_isMu',
      'pfcand_thetarel',
      'pfcand_phirel'
    ]

    # read input file
    print('Reading input file...')
    jets = read_file(args.inputfile, treename='tree', branches=branches_to_read)
    print(f'Found {len(jets)} jets.')

    # do upsampling
    print('Doing upsampling...')
    copy_branches = [b for b in branches_to_read if not b.startswith('pfcand_')]
    jets_upsampled = upsample_naive(jets, 
                       factor=args.upsample_factor,
                       std=args.std,
                       copy_branches=copy_branches)
    print(f'Created {len(jets_upsampled)} upsampled jets.')
    
    # write output file
    write_tree(jets_upsampled, args.outputfile, treename='tree',
      records=['pfcand'], counters_to_remove=['npfcand*', 'nrecojet*'])
    print(f'Output file {args.outputfile} written.')
