# Copy samples and split into smaller parts

# Use case: background samples are too big for practical purposes,
# would be easier to split them into many smaller files.

# Note: does not work yet, unresolved dtype errors on writing.

import os
import sys
import uproot
import awkward as ak


if __name__=='__main__':

    # set sources and targets
    cpdirs = ({
        '/blue/avery/ekoenig/Run3_HHTo4B_NTuples/2021/Main_PNet_MinDiag_w4j35_w2bj30_dHHjw30_AN23151_PNetRegOnLooseJets_StandardForOthers_woSyst_18Nov2024_2021_0L/mc/':
        '/blue/avery/llambre1.brown/hhto4b-samples/ntuples-bkg/ntuples_2021',
        '/blue/avery/ekoenig/Run3_HHTo4B_NTuples/2022/Main_PNet_MinDiag_w4j35_w2bj30_dHHjw30_AN23151_PNetRegOnLooseJets_StandardForOthers_woSyst_18Nov2024_2022_0L/mc/':
        '/blue/avery/llambre1.brown/hhto4b-samples/ntuples-bkg/ntuples_2022',
        '/blue/avery/ekoenig/Run3_HHTo4B_NTuples/2023/Main_PNet_MinDiag_4j30_2bj30_dHHjw25_AN23151_PNetRegOnLooseJets_StandardForOthers_woSyst_18Nov2024_2023_0L/mc/':
        '/blue/avery/llambre1.brown/hhto4b-samples/ntuples-bkg/ntuples_2023',
        '/blue/avery/ekoenig/Run3_HHTo4B_NTuples/2020/Main_PNet_MinDiag_4j30_2bj30_dHHjw25_AN23151_PNetRegOnLooseJets_StandardForOthers_woSyst_18Nov2024_2020_0L/mc/':
        '/blue/avery/llambre1.brown/hhto4b-samples/ntuples-bkg/ntuples_2020'
    })
    cpmap = {}
    for sourcedir, targetdir in cpdirs.items():
        for f in ['qcd-mg_tree.root', 'ttbar-powheg_tree.root']:
            cpmap[os.path.join(sourcedir, f)] = os.path.join(targetdir, f)

    # other settings
    treename = 'Events'
    events_per_file = 10000
    max_files = 10

    # file-specific settings
    # (did not find an elegant way to avoid this...)
    recordnames = ['ak4', 'ak8']

    # loop over files to copy
    for idx, (source, target) in enumerate(cpmap.items()):
        print('Now running on file {} ({}/{})'.format(source, idx+1, len(cpmap)))
    
        # make target dir
        targetdir = os.path.dirname(target)
        if not os.path.exists(targetdir): os.makedirs(targetdir)

        # read file
        with uproot.open(source) as f:
            tree = f[treename]
            print('Read tree with {} events'.format(tree.num_entries))

            # loop over batches
            for batchidx in range(max_files):

                # break the loop if end of the source file was reached
                if batchidx * events_per_file > tree.num_entries: break

                # set start and stop entry number
                start = batchidx * events_per_file
                stop = (batchidx+1) * events_per_file

                # read the branches
                branches = tree.arrays(entry_start=start, entry_stop=stop)
                print('Read batch {} with {} entries and {} branches'.format(
                    batchidx, len(branches), len(branches.fields)))

                # convert to format suitable for writing
                # (see here: https://github.com/scikit-hep/uproot5/discussions/903)
                writebranches = dict(zip(ak.fields(branches), ak.unzip(branches)))
                for recordname in recordnames:
                    tag = recordname + '_'
                    writebranches = {k: v for k, v in writebranches.items() if not k.startswith(tag)}
                    record = ak.zip({name[len(tag):]: array for name, array in zip(ak.fields(branches), ak.unzip(branches)) if name.startswith(tag)})
                    writebranches[recordname] = record
                
                # need to remove counter branches as they are added automatically,
                # otherwise gives strange dtype errors...
                writebranches = {k: v for k, v in writebranches.items() if not (k[0]=='n' and k[1].isupper())}

                # write to file
                batchfile = target.replace('.root', f'_{batchidx}.root')
                with uproot.recreate(batchfile) as outf:
                    outf[treename] = writebranches
                print(f'Batch written to output file {batchfile}')

