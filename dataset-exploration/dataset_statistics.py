# Print statistics of a dataset of ntuples

# Counts the number of files, size on disk, and number of jets
# (in total, per class, and passing the jet selection),
# grouped per subdirectory of the input directory.
# Intended for the per-jet ntuples in
# /eos/experiment/fcc/ee/analyses/case-studies/aleph/processedMC/1994/zqq/stage2,
# but also works on the per-event ntuples in the corresponding stage1 directory
# (use --perevent in that case).

import os
import sys
import glob
import json
import argparse
import collections
import numpy as np
import awkward as ak
import uproot
from concurrent.futures import ProcessPoolExecutor


default_inputdir = '/eos/experiment/fcc/ee/analyses/case-studies/aleph/processedMC/1994/zqq/stage2/VTAG-01'
label_branches = ['recojet_isB', 'recojet_isC', 'recojet_isS', 'recojet_isU', 'recojet_isD']
pid_names = {5: 'B', 4: 'C', 3: 'S', 2: 'U', 1: 'D'}


def count_file(args):
    ### count the jets in a single file
    # returns a dict of counters (or a dict with an 'error' key if the file could not be read)
    (path, treename, perevent, ptmin, etamax) = args
    res = {'path': path, 'size': os.path.getsize(path)}
    try:
        with uproot.open(path) as f:
            tree = f[treename]
            ncycles = len([k for k in f.keys() if k.split(';')[0] == treename])
            res['ncycles'] = ncycles
            if perevent:
                # one entry per event, jets stored as vectors;
                # no label branches, the flavour is given by jetPID (per event)
                arrays = tree.arrays(['jet_pT', 'jet_eta', 'jetPID'], library='ak')
                njets = ak.num(arrays['jet_pT'])
                sel = (arrays['jet_pT'] > ptmin) & (abs(arrays['jet_eta']) < etamax)
                res['nevents'] = int(tree.num_entries)
                res['njets'] = int(ak.sum(njets))
                res['njets_sel'] = int(ak.sum(sel))
                pid = ak.to_numpy(arrays['jetPID']).astype(int)
                njets = ak.to_numpy(njets)
                nsel = ak.to_numpy(ak.sum(sel, axis=1))
                for val, name in pid_names.items():
                    res[f'njets_{name}'] = int(njets[pid == val].sum())
                    res[f'njets_sel_{name}'] = int(nsel[pid == val].sum())
                res['nunlabeled'] = int(njets[~np.isin(pid, list(pid_names.keys()))].sum())
            else:
                # one entry per jet
                arrays = tree.arrays(label_branches + ['jet_pT', 'jet_eta'], library='np')
                sel = (arrays['jet_pT'] > ptmin) & (np.abs(arrays['jet_eta']) < etamax)
                res['njets'] = int(tree.num_entries)
                res['njets_sel'] = int(sel.sum())
                for branch in label_branches:
                    name = branch.replace('recojet_is', '')
                    res[f'njets_{name}'] = int(arrays[branch].sum())
                    res[f'njets_sel_{name}'] = int(arrays[branch][sel].sum())
                labelsum = np.sum([arrays[branch] for branch in label_branches], axis=0)
                res['nunlabeled'] = int((labelsum != 1).sum())
    except Exception as e:
        res['error'] = repr(e)
    return res


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', default=default_inputdir,
      help='Top-level directory, all .root files below it are counted.')
    parser.add_argument('-t', '--treename', default=None,
      help='Tree name (default: "tree" for per-jet ntuples, "events" for per-event ntuples).')
    parser.add_argument('--perevent', default=False, action='store_true',
      help='Input files are per-event ntuples (stage1) instead of per-jet ntuples (stage2).')
    parser.add_argument('--ptmin', default=10., type=float,
      help='Jet pt threshold of the jet selection (as in the data config).')
    parser.add_argument('--etamax', default=0.9, type=float,
      help='Jet abs(eta) threshold of the jet selection (as in the data config).')
    parser.add_argument('-n', '--nworkers', default=8, type=int)
    parser.add_argument('-o', '--outputfile', default=None,
      help='Optional .json file to store the per-file counts.')
    args = parser.parse_args()
    if args.treename is None: args.treename = 'events' if args.perevent else 'tree'

    # find input files
    inputdir = os.path.abspath(args.inputdir)
    files = sorted(glob.glob(os.path.join(inputdir, '**', '*.root'), recursive=True))
    print(f'Found {len(files)} .root files in {inputdir}.')
    if len(files)==0: sys.exit()

    # count all files
    jobs = [(f, args.treename, args.perevent, args.ptmin, args.etamax) for f in files]
    results = []
    with ProcessPoolExecutor(args.nworkers) as executor:
        for idx, res in enumerate(executor.map(count_file, jobs, chunksize=2)):
            results.append(res)
            if (idx+1) % 100 == 0: print(f'  Processed {idx+1} / {len(files)} files...', flush=True)
    if args.outputfile is not None:
        with open(args.outputfile, 'w') as f: json.dump(results, f, indent=1)
        print(f'Per-file counts written to {args.outputfile}.')

    # report files that could not be read
    errors = [res for res in results if 'error' in res]
    print(f'Number of files that could not be read: {len(errors)}')
    for res in errors: print(f'  - {res["path"]}: {res["error"]}')

    # aggregate per subdirectory
    counts = collections.defaultdict(collections.Counter)
    for res in results:
        if 'error' in res: continue
        group = os.path.relpath(os.path.dirname(res['path']), inputdir)
        for key in [group, 'TOTAL']:
            counts[key]['nfiles'] += 1
            counts[key]['multiple_cycles'] += int(res['ncycles'] > 1)
            for field, val in res.items():
                if field in ['path', 'ncycles']: continue
                counts[key][field] += val

    # print table
    names = list(pid_names.values())
    columns = ['nfiles', 'size (GB)']
    if args.perevent: columns.append('nevents')
    columns += ['njets'] + [f'njets_{name}' for name in names] + ['njets_sel']
    columns += [f'njets_sel_{name}' for name in names] + ['nunlabeled', 'multiple_cycles']
    width = max(len(key) for key in counts.keys())
    print()
    print(f'Jet selection: pt > {args.ptmin} and abs(eta) < {args.etamax}')
    print(' | '.join(['group'.ljust(width)] + [c.rjust(11) for c in columns]))
    for key in sorted(counts.keys(), key=lambda k: (k=='TOTAL', k)):
        row = [key.ljust(width)]
        for column in columns:
            if column=='size (GB)': row.append(f'{counts[key]["size"]/1e9:.1f}'.rjust(11))
            else: row.append(str(counts[key][column]).rjust(max(11, len(column))))
        print(' | '.join(row))
