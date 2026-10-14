# Compare the model evaluation for jet flavour tagging between multiple trainings

# Makes the same ROC curves as evaluate_bcslight.py,
# but with one figure per pair of categories (e.g. b-jets vs c-jets),
# each comparing the results from multiple input files (e.g. different trainings).

import os
import sys
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import roc_auc_score

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from tools import read_file


def calculate_roc(events, signal_category, background_category):
    ### calculate the ROC curve and AUC for a pair of categories
    # (same method as in plot_roc_multi.py)
    # returns a tuple (signal efficiency, background efficiency, AUC),
    # or None if there are no signal or background events.

    # get scores for signal and background
    # (note: just use the signal score, no binarization)
    scores = events[signal_category['score_branch']]
    scores_sig = scores[events[signal_category['label_branch']].astype(bool)]
    scores_bkg = scores[events[background_category['label_branch']].astype(bool)]
    if len(scores_sig)==0 or len(scores_bkg)==0: return None

    # calculate AUC
    this_scores = np.concatenate((scores_sig, scores_bkg))
    this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
    auc = roc_auc_score(this_labels, this_scores)

    # calculate signal and background efficiency
    thresholds = np.concatenate((
        np.linspace(np.amin(this_scores), np.amax(this_scores)*0.9, num=100),
        np.linspace(np.amax(this_scores)*0.9, np.amax(this_scores)*0.98, num=300),
        np.linspace(np.amax(this_scores)*0.98, np.amax(this_scores), num=300)
    ))
    efficiency_sig = np.array([np.sum(scores_sig > threshold) for threshold in thresholds]) / len(scores_sig)
    efficiency_bkg = np.array([np.sum(scores_bkg > threshold) for threshold in thresholds]) / len(scores_bkg)
    return (efficiency_sig, efficiency_bkg, auc)


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', required=True, nargs='+',
      help='Output directories of the trainings (containing output.root), or the root files themselves.')
    parser.add_argument('-l', '--labels', required=True, nargs='+',
      help='Legend entries, in the same order as the inputs.')
    parser.add_argument('-o', '--outputdir', required=True,
      help='Directory where to store the figures.')
    args = parser.parse_args()

    # check arguments
    if len(args.labels) != len(args.inputs):
        msg = f'Number of labels ({len(args.labels)}) does not match number of inputs ({len(args.inputs)}).'
        raise Exception(msg)
    inputfiles = [os.path.join(f, 'output.root') if os.path.isdir(f) else f for f in args.inputs]
    for f in inputfiles:
        if not os.path.exists(f): raise Exception(f'File {f} does not exist.')

    # hard coded settings
    treename = 'Events'
    categories = {
        'b': {
            'label_branch': 'recojet_isB',
            'score_branch': 'score_recojet_isB',
            'label': r'$b$-jets'
        },
        'c': {
            'label_branch': 'recojet_isC',
            'score_branch': 'score_recojet_isC',
            'label': r'$c$-jets'
        },
        's': {
            'label_branch': 'recojet_isS',
            'score_branch': 'score_recojet_isS',
            'label': r'$s$-jets'
        },
        'ud': {
            'label_branch': 'recojet_isUDG',
            'score_branch': 'score_recojet_isUDG',
            'label': r'$ud$-jets'
        }
    }
    sig_effs = [0.2, 0.4, 0.6, 0.8]

    # find all branches to read
    branches_to_read = ([cat['label_branch'] for cat in categories.values()]
                        + [cat['score_branch'] for cat in categories.values()])

    # calculate the ROC curves for each input file and each pair of categories
    pairs = [(sig, bkg) for sidx, sig in enumerate(categories) for bidx, bkg in enumerate(categories) if bidx > sidx]
    rocs = {pair: [] for pair in pairs}
    for inputfile in inputfiles:
        print(f'Running on input file {inputfile}...')
        events = read_file(inputfile, treename=treename, branches=branches_to_read)
        print(f'  Number of events: {len(events[branches_to_read[0]])}')
        for pair in pairs:
            rocs[pair].append(calculate_roc(events, categories[pair[0]], categories[pair[1]]))

    # make output directory
    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

    # make a figure for each pair of categories
    cmap = plt.get_cmap('cool', len(inputfiles))
    table = {}
    for pair in pairs:
        pair_label = categories[pair[0]]['label'] + ' vs. ' + categories[pair[1]]['label']
        pair_name = f'{pair[0]}_vs_{pair[1]}'
        fig, ax = plt.subplots()

        # plot the ROC curve for each input file
        table[pair_name] = {'sig_effs': sig_effs}
        for idx, (roc, label) in enumerate(zip(rocs[pair], args.labels)):
            if roc is None: continue
            (efficiency_sig, efficiency_bkg, auc) = roc
            ax.plot(efficiency_bkg, efficiency_sig,
              color=cmap(idx), linewidth=3, label=label + ' (AUC: {:.3f})'.format(auc))
            # make a table entry (background pass-through at given signal efficiencies)
            table_entry = []
            for sig_eff in sig_effs:
                bidx = np.nonzero(efficiency_sig[::-1] > sig_eff)[0][0]
                table_entry.append(float(efficiency_bkg[::-1][bidx]))
            table[pair_name][label] = table_entry

        # add random guessing line
        dummy_efficiency = np.linspace(0, 1, num=101)
        ax.plot(dummy_efficiency, dummy_efficiency,
          color='darkblue', linewidth=3, linestyle='--', label='Random guessing')

        # other plot settings
        ax.set_xlabel('Background pass-through', fontsize=15)
        ax.set_ylabel('Signal efficiency', fontsize=15)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(which='both')
        ax.set_ylim((-0.05, 1.1))
        leg = ax.legend(loc='lower right')
        # write the type of curve in the upper left corner, in the same style as the legend
        # (same box, padding and distance to the axes)
        frame = leg.get_frame()
        textbox = AnchoredText(pair_label, loc='upper left',
          prop={'fontsize': leg.get_texts()[0].get_fontsize()},
          pad=leg.borderpad, borderpad=leg.borderaxespad)
        textbox.patch.set_boxstyle(frame.get_boxstyle())
        textbox.patch.set_facecolor(frame.get_facecolor())
        textbox.patch.set_edgecolor(frame.get_edgecolor())
        textbox.patch.set_alpha(frame.get_alpha())
        ax.add_artist(textbox)

        # save figure
        fig.tight_layout()
        figname = os.path.join(args.outputdir, f'roc_{pair_name}.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')

        # same with log scale on x-axis
        ax.set_xscale('log')
        ax.set_xlim((1e-6, 1))
        fig.tight_layout()
        figname = os.path.join(args.outputdir, f'roc_{pair_name}_log.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
        plt.close()

    # print and store the table
    print('Background pass-through at signal efficiencies {}:'.format(sig_effs))
    for pair_name, entries in table.items():
        print(f'  {pair_name}:')
        for label, values in entries.items():
            if label=='sig_effs': continue
            print('    {0: <30}'.format(label) + ''.join(['{0: <12}'.format('{:.3e}'.format(v)) for v in values]))
    filename = os.path.join(args.outputdir, 'table.json')
    with open(filename, 'w') as f:
        json.dump(table, f, indent=2)
    print(f'Saved table {filename}.')
