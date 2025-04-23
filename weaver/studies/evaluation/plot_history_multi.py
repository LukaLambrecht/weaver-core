# Plot training (and validation) history for multiple networks for comparison

import os
import sys
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_history_multi(histories):
    '''
    Input arguments:
    - histories: a dict of history objects, see plot_history for more info.
    '''

    # convert list of dicts to global axes
    values = {}
    history_names = list(histories.keys())
    metric_names = list(histories[history_names[0]][0].keys())
    for history_name, history in histories.items():
        values[history_name] = {}
        for metric_name in history[0].keys():
            values[history_name][metric_name] = np.array([epoch[metric_name][idx]
                for epoch in history for idx in range(len(epoch[metric_name]))])
    xax = np.arange(len(values[history_names[0]][metric_names[0]]))

    # loop over histories and metrics
    figsize = (6, 6 + 2*(len(metric_names)-1))
    fig, axs = plt.subplots(nrows=len(metric_names), figsize=figsize)
    for axidx, metric_name in enumerate(metric_names):
        ax = axs[axidx]
        cmap = plt.get_cmap('cool', len(history_names))
        for hidx, history_name in enumerate(history_names):
            ax.plot(xax, values[history_name][metric_name],
                    alpha=0.5, color=cmap(hidx), label=history_name)
        ax.set_xlabel('Batches', fontsize=12)
        ax.set_ylabel(f'Loss ({metric_name})', fontsize=12)
        ax.grid()
        ax.set_yscale('log')
    
    leg = axs[0].legend(fontsize=10)
    fig.tight_layout()
    return fig, axs
