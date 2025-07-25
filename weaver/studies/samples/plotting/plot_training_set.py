# Make plots of the training set

# Use cases:
# - checking if reweighting works as expected

# Note: originally copied from the main weaver script train.py,
# but removing all unnecessary stuff and adding the plotting part


import os
import ast
import sys
import shutil
import glob
import json
import argparse
import functools
import numpy as np
import math
import copy
import torch
import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from weaver.utils.logger import _logger
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.samplelisttools import read_sample_list
from weaver.train import parse_file_patterns


# handle the logger, which is implicitly called by many helper functions,
# but only works properly if running the 'weaver' executable...
# so redirect to simple print
def print_wrapper(text, *moretext, **kwargs):
    toprint = [str(el) for el in moretext]
    print(' '.join([text] + toprint))
_logger.info = print_wrapper
_logger.error = print_wrapper


# read command line args
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--data-config', required=True, type=str,
                    help='data config YAML file')
parser.add_argument('-i', '--data-train', required=True, nargs='*',
                    help='sample list YAML file')
parser.add_argument('-o', '--outputdir', required=True,
                    help='output directory for plots')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file')
parser.add_argument('--in-memory', action='store_true', default=False,
                    help='load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run')
parser.add_argument('--copy-inputs', action='store_true', default=False,
                    help='copy input files to the current dir (can help to speed up dataloading when running over remote files, e.g., from EOS)')


def load_data(args):
    """
    Loads the data for plotting.
    Input arguments:
     - full set of command line args
    """

    # get the file patterns in the case of a provided sample list
    if len(args.data_train)==1 and args.data_train[0].endswith('.yaml'):
        samplelist = args.data_train[0]
        print(f'Reading sample list {samplelist} for training data.')
        args.data_train = read_sample_list(samplelist)

    # get the files
    train_file_dict, train_files = parse_file_patterns(args.data_train,
        copy_inputs=args.copy_inputs, local_rank=None)

    # print number of files for debugging
    print('Using %d files for training' % (len(train_files)))

    # make training data loader
    name = 'train'
    train_data = SimpleIterDataset(train_file_dict, args.data_config,
                                   for_training=True,
                                   remake_weights=True,
                                   load_range_and_fraction=((0, 1), args.data_fraction),
                                   fetch_by_files=False,
                                   fetch_step=1,
                                   in_memory=args.in_memory,
                                   name=name)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                     drop_last=False, pin_memory=True,
                     num_workers=0, persistent_workers=False)

    # make a new reference to the data config
    data_config = train_data.config
    train_input_names = train_data.config.input_names
    train_label_names = train_data.config.label_names

    # return data loaders and some other info
    return train_loader, data_config, train_input_names, train_label_names


def _main(args):

    # print all arguments
    print('Running weaver (train.py) with following arguments:')
    print('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    # define variables
    # (hard-coded for now, maybe extend later)
    variables = [
        {
            'name': 'dHH_H1_mass',
            'variable': 'dHH_H1_mass',
            'bins': [-1, 0, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 30000],
            'label': 'dHH_H1_mass'
        },
        {
            'name': 'dHH_H2_mass',
            'variable': 'dHH_H2_mass',
            'bins': [-1, 0, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 30000],
            'label': 'dHH_H2_mass'
        },
        {
            'name': 'hh_average_mass',
            'variable': 'hh_average_mass',
            'bins': [-1, 0, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 30000],
            'label': 'hh_average_mass'
        },
    ]

    # define processes
    # (hard-coded for now, maybe extend later)
    processes = [
        {
            'name': 'sig',
            'variable': 'isSignal',
            'label': 'Sig.',
            'color': 'dodgerblue'
        },
        {
            'name': 'bkg',
            'variable': 'isBackground',
            'label': 'Bkg.',
            'color': 'darkorchid'
        }
    ]

    # define other parameters
    # (hard-coded for now, maybe extend later)
    batch_size = 5000
    num_batches = 10

    # load data
    print('Loading training data...')
    args.batch_size = batch_size
    train_loader, data_config, train_input_names, train_label_names = load_data(args)
    print('Done loading training data.')

    # get variable values for each process
    varvalues = {}
    for process in processes:
        varvalues[process['name']] = {variable['variable']: [] for variable in variables}
    batch_count = 0
    for X, y, Z in train_loader:
            if batch_count >= num_batches: break
            print(f'Batch {batch_count+1}...')
            batch_count += 1
            masks = {}
            for process in processes:
                mask = Z[process['variable']].numpy().astype(bool)
                masks[process['name']] = mask
                print(f'Process {process["name"]}: {np.sum(mask)} entries')
            for variable in variables:
                values = Z[variable['variable']].numpy()
                for process in processes:
                    batch = values[masks[process['name']]]
                    varvalues[process['name']][variable['variable']].append(batch)

    # concatenate
    for process in processes:
        for variable in variables:
            batches = varvalues[process['name']][variable['variable']]
            varvalues[process['name']][variable['variable']] = np.concatenate(tuple(batches))

    # make histograms
    hists = {}
    for process in processes:
        hists[process['name']] = {}
        for variable in variables:
            values = varvalues[process['name']][variable['variable']]
            weights = np.ones(len(values))
            hist = np.histogram(values, bins=variable['bins'], weights=weights)[0]
            staterror = np.sqrt(np.histogram(values, bins=variable['bins'],
                      weights=np.square(weights))[0])
            hists[process['name']][variable['name']] = (hist, staterror)

    # make output directory
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # make plots
    for variable in variables:
        fig, ax = plt.subplots()
        for process in processes:
            hist, staterrors = hists[process['name']][variable['name']]
            # remove under- and overflow bins
            bins = variable['bins'][1:-1]
            hist = hist[1:-1]
            staterrors = staterrors[1:-1]
            ax.stairs(hist, edges=bins,
                  color = process['color'],
                  label = process['label'],
                  linewidth=2)
            ax.stairs((hist+staterrors), baseline=(hist-staterrors),
                    edges=bins,
                    color = process['color'], fill=True, alpha=0.15)
        ax.set_xlabel(variable['label'], fontsize=12)
        ax.set_ylabel('Entries', fontsize=12)
        leg = ax.legend(fontsize=12)
        ax.set_ylim((0, 1.4*ax.get_ylim()[1]))
        ax.text(0.05, 0.95, f'Batch size: {batch_size}',
                va='top', ha='left', transform=ax.transAxes,
                fontsize=10)
        ax.text(0.05, 0.9, f'Number of batches: {num_batches}',
                va='top', ha='left', transform=ax.transAxes,
                fontsize=10)
        fig.tight_layout()
        figname = os.path.join(args.outputdir, variable['name']+'.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
        plt.close()
            

def main():

    # parse command line args
    args = parser.parse_args()
    _main(args)


if __name__ == '__main__':
    main()
