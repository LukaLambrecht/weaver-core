# Plot training (and validation) history

import os
import sys
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_history(history, colordict=None):
    '''
    Input arguments:
    - history: a list of dicts, each dict represents a single epoch
      and has the following form: {'some metric': [values over batches in this epoch], ...}
    '''

    # convert list of dicts to global axes
    values = {}
    for key in history[0].keys():
        values[key] = np.array([epoch[key][idx] for epoch in history for idx in range(len(epoch[key]))])
    xax = np.arange(len(values[list(values.keys())[0]]))

    # make the base figure
    fig, ax = plt.subplots()
    for key, val in values.items():
        color = colordict.get(key, 'grey') if colordict is not None else None
        linewidth = 2 if key.lower()=='total' else 1
        ax.plot(xax, val, alpha=0.5,
                color=color, label=key, linewidth=linewidth)

    # plot aesthetics
    ax.set_xlabel('Batches', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid()
    ax.set_yscale('log')
    leg = ax.legend(fontsize=10)
    fig.tight_layout()
    return fig, ax

if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True, nargs='+')
    args = parser.parse_args()

    # set color dict (hard-coded)
    colordict = {
      'total': 'r',
      'BCE': 'mediumblue',
      'DisCo': 'darkviolet'
    }

    # loop over input files
    for inputfile in args.inputfile:
        
        # load json object
        with open(inputfile, 'r') as f:
            history = json.load(f)

        # make plot
        fig, ax = plot_history(history,
                    colordict=colordict)

        # save fig
        outputfile = inputfile.replace('.json', '.png')
        fig.savefig(outputfile)
        print(f'Saved figure {outputfile}')
