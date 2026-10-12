# Compare the new per-jet ntuples to the old ones

# Checks the following:
# - which branches used in a data config are missing or renamed in the new ntuples.
# - whether the distributions of the common branches agree
#   (by comparing quantiles and the fractions of default values).
# The old ntuples are assumed to contain all flavours (in natural proportions) in each file,
# while the new ntuples are assumed to be split in one directory per flavour;
# jets from the different flavours are mixed in natural proportions for the comparison.

import os
import re
import sys
import glob
import argparse
import numpy as np
import awkward as ak
import uproot


thisdir = os.path.abspath(os.path.dirname(__file__))
default_old = '/eos/user/l/llambrec/aleph-data/ntuples-withksnew/jetlevel/mc/output_qqb_10_train.root'
default_new = '/eos/experiment/fcc/ee/analyses/case-studies/aleph/processedMC/1994/zqq/stage2/VTAG-01'
default_config = os.path.join(thisdir, '../weaver/configs/configs_parttaggerwithv0/standardized',
                   'data_config_parttaggerwithv0_extrafeatures_test.yaml')

# branch names that are different in the new ntuples (old name -> new name)
# (on top of the generic replacement v0cand_ -> v0_)
# note: the correspondence of sv/v0 xrel/yrel/zrel with dx/dy/dz is a guess based on similar scale!
renamed_branches = {
    'recojet_pt': 'jet_pT',
    'recojet_eta': 'jet_eta',
    'recojet_theta': 'jet_theta',
    'recojet_phi': 'jet_phi',
    'recojet_e': 'jet_e',
    'recojet_mass': 'jet_mass',
    'nconst': 'jet_nconst',
    'npfcand': 'n_pfcand',
    'nsv': 'n_sv',
    'nv0cand': 'n_v0',
    'recojet_nkscandidates': 'n_v0_ks',
    'recojet_nlambdacandidates': 'n_v0_lambda',
    'nchargedhad': 'jet_nchad',
    'nneutralhad': 'jet_nnhad',
    'nphotons': 'jet_ngamma',
    'nel': 'jet_nel',
    'nmu': 'jet_nmu',
    'pfcand_Sip2dVal': 'pfcand_btagSip2dVal',
    'pfcand_Sip2dSig': 'pfcand_btagSip2dSig',
    'pfcand_Sip3dVal': 'pfcand_btagSip3dVal',
    'pfcand_Sip3dSig': 'pfcand_btagSip3dSig',
    'pfcand_JetDistVal': 'pfcand_btagJetDistVal',
    'pfcand_JetDistSig': 'pfcand_btagJetDistSig',
    'pfcand_trackChi2Normalized': 'pfcand_trackChi2Norm',
    'sv_chi2Normalized': 'sv_chi2_norm',
    'sv_nTracks': 'sv_ntracks',
    'sv_xrel': 'sv_dx',
    'sv_yrel': 'sv_dy',
    'sv_zrel': 'sv_dz',
    'v0cand_mass': 'v0_invM',
    'v0cand_pdgId': 'v0_pdg',
    'v0cand_chi2Normalized': 'v0_chi2_norm',
    'v0cand_nTracks': 'v0_ntracks',
    'v0cand_xrel': 'v0_dx',
    'v0cand_yrel': 'v0_dy',
    'v0cand_zrel': 'v0_dz',
}

# branches to compare on top of the ones used in the data config
extra_branches = ['npfcand', 'nsv', 'nv0cand', 'recojet_nkscandidates', 'recojet_nlambdacandidates',
                  'v0cand_pdgId', 'pfcand_erel', 'pfcand_ptrel', 'pfcand_theta', 'pfcand_phi', 'sv_ndof']

# flavour subdirectories of the new ntuples with their natural fractions in Z -> qq
flavour_fractions = {'Zbb': 0.216, 'Zcc': 0.172, 'Zss': 0.222, 'Zuu': 0.160, 'Zdd': 0.222}

quantiles = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]


def new_name(branch):
    ### get the name of a branch in the new ntuples
    return renamed_branches.get(branch, branch.replace('v0cand_', 'v0_'))

def summarize(array):
    ### get summary statistics of a (potentially jagged) array
    values = ak.to_numpy(ak.flatten(array, axis=None)).astype(float)
    if len(values)==0: return None
    finite = np.isfinite(values)
    return {
        'quantiles': np.quantile(values[finite], quantiles),
        'mean': values[finite].mean(),
        'std': values[finite].std(),
        'nonfinite': (~finite).mean(),
        'default': (values==-9).mean(),
        'zero': (values==0).mean()
    }

def format_summary(summary):
    if summary is None: return 'EMPTY'
    txt = ' '.join(f'{val:10.4g}' for val in summary['quantiles'])
    txt += f' | mean {summary["mean"]:10.4g} std {summary["std"]:9.4g}'
    txt += f' nonfinite {summary["nonfinite"]:.3f} ==-9 {summary["default"]:.3f} ==0 {summary["zero"]:.3f}'
    return txt

def find_differences(old, new, quantile_threshold=0.15, fraction_threshold=0.03):
    ### list the differences between two summaries
    if old is None or new is None: return ['empty']
    differences = []
    # shift of the quantiles (excluding min and max) relative to the 1% - 99% range
    scale = max(abs(old['quantiles'][-2] - old['quantiles'][1]),
                abs(new['quantiles'][-2] - new['quantiles'][1]), 1e-12)
    shift = np.max(np.abs(old['quantiles'][1:-1] - new['quantiles'][1:-1])) / scale
    if shift > quantile_threshold: differences.append(f'quantile shift {shift:.2f}')
    # fractions of special values
    for key, label in [('default', '==-9'), ('zero', '==0'), ('nonfinite', 'nonfinite')]:
        if abs(old[key] - new[key]) > fraction_threshold:
            differences.append(f'fraction {label}: {old[key]:.3f} (old) vs {new[key]:.3f} (new)')
    return differences


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', default=default_old,
      help='File (or file pattern, first match is used) with old ntuples.')
    parser.add_argument('--new', default=default_new,
      help='Directory with new ntuples (containing one subdirectory per flavour).')
    parser.add_argument('--new_subdir', default='QQB/train',
      help='Subdirectory inside each flavour directory of the new ntuples.')
    parser.add_argument('--new_fileidx', default=5, type=int,
      help='Index of the file (in sorted order) to read for each flavour of the new ntuples.')
    parser.add_argument('-c', '--config', default=default_config,
      help='Data config, used to find the branches that are needed.')
    parser.add_argument('-t', '--treename', default='tree')
    parser.add_argument('-n', '--njets', default=40000, type=int,
      help='Number of jets to read from both the old and the new ntuples (before selection).')
    parser.add_argument('--ptmin', default=10., type=float)
    parser.add_argument('--etamax', default=0.9, type=float)
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
      help='Print the summary statistics of all branches instead of only the differing ones.')
    args = parser.parse_args()

    # open the files
    oldfile = sorted(glob.glob(args.old))[0]
    newfiles = {}
    for flavour in flavour_fractions.keys():
        pattern = os.path.join(args.new, flavour, args.new_subdir, '*.root')
        newfiles[flavour] = sorted(glob.glob(pattern))[args.new_fileidx]
    print(f'Old file: {oldfile}')
    print('New files:')
    for f in newfiles.values(): print(f'  - {f}')
    oldtree = uproot.open(oldfile)[args.treename]
    newtrees = {flavour: uproot.open(f)[args.treename] for flavour, f in newfiles.items()}
    oldbranches = oldtree.keys()
    newbranches = set(newtrees['Zbb'].keys())

    # check which branches used in the data config are available in the new ntuples
    with open(args.config) as f: config = f.read()
    needed = sorted([b for b in oldbranches if re.search(r'\b{}\b'.format(b), config)])
    missing = [b for b in needed if new_name(b) not in newbranches]
    renamed = {b: new_name(b) for b in needed if new_name(b) in newbranches and new_name(b)!=b}
    print()
    print(f'Data config: {os.path.abspath(args.config)}')
    print(f'Number of branches in the old ntuples used in the data config: {len(needed)}')
    print(f'Branches missing in the new ntuples ({len(missing)}):')
    for b in missing: print(f'  - {b}')
    print(f'Branches renamed in the new ntuples ({len(renamed)}):')
    for b, n in renamed.items(): print(f'  - {b} -> {n}')
    print('Branches in the new ntuples without counterpart in the old ntuples:')
    print('  ' + ' '.join(sorted(newbranches - set(new_name(b) for b in oldbranches))))
    print('Branches in the old ntuples without counterpart in the new ntuples:')
    print('  ' + ' '.join(sorted(b for b in oldbranches if new_name(b) not in newbranches)))

    # load the jets
    branches = [b for b in needed + extra_branches if b in oldbranches and new_name(b) in newbranches]
    old = oldtree.arrays(branches, entry_stop=args.njets)
    new = ak.concatenate([
            newtrees[flavour].arrays([new_name(b) for b in branches], entry_stop=int(args.njets*frac))
            for flavour, frac in flavour_fractions.items()])
    old = old[(old['recojet_pt'] > args.ptmin) & (abs(old['recojet_eta']) < args.etamax)]
    new = new[(new['jet_pT'] > args.ptmin) & (abs(new['jet_eta']) < args.etamax)]
    print()
    print(f'Number of jets after selection: {len(old)} (old), {len(new)} (new)')

    # compare the lengths of the collections
    print('Maximum number of objects per jet (old / new):')
    for name, branch in [('pfcand', 'pfcand_pt'), ('sv', 'sv_mass'), ('v0', 'v0cand_mass')]:
        print(f'  - {name}: {ak.max(ak.num(old[branch]))} / {ak.max(ak.num(new[new_name(branch)]))}')
    print('V0 PDG IDs (values, counts):')
    for label, values in [('old', old['v0cand_pdgId']), ('new', new['v0_pdg'])]:
        print(f'  - {label}:', *np.unique(ak.to_numpy(ak.flatten(values)).astype(int), return_counts=True))

    # compare the distributions
    header = ' '*37 + ' '.join(f'{f"{int(q*100)}%":>10s}' for q in quantiles)
    consistent = []
    print()
    print('Comparison of distributions (O = old, N = new):')
    print(header)
    for branch in branches:
        oldsummary = summarize(old[branch])
        newsummary = summarize(new[new_name(branch)])
        differences = find_differences(oldsummary, newsummary)
        if len(differences)==0: consistent.append(branch)
        if len(differences)==0 and not args.verbose: continue
        name = branch if new_name(branch)==branch else f'{branch} -> {new_name(branch)}'
        print(f'{name}: ' + ('; '.join(differences) if len(differences)>0 else 'consistent'))
        print(f'{"":34s} O {format_summary(oldsummary)}')
        print(f'{"":34s} N {format_summary(newsummary)}')
    print()
    print(f'Branches with consistent distributions ({len(consistent)} / {len(branches)}):')
    print('  ' + ' '.join(consistent))
