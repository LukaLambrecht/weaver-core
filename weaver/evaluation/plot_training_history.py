import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt


def parse_info(json_info):
    '''
    Parse json-formatted info
    '''

    # get epochs
    epochs = json_info["epochs"]
    nepochs = len(epochs)

    # initialize arrays
    train_loss = np.zeros(nepochs)
    val_loss = np.zeros(nepochs)
    train_acc = np.zeros(nepochs)
    val_acc = np.zeros(nepochs)
    ninstances = np.zeros(nepochs)

    # loop over epochs and gather values
    for epoch_idx in range(nepochs):
        train_loss[epoch_idx] = epochs[epoch_idx]["train"]["loss"]
        train_acc[epoch_idx] = epochs[epoch_idx]["train"]["accuracy"]
        val_loss[epoch_idx] = epochs[epoch_idx]["validation"]["loss"]
        val_acc[epoch_idx] = epochs[epoch_idx]["validation"]["accuracy"]
        ninstances_base = 0 if epoch_idx==0 else ninstances[epoch_idx-1]
        ninstances[epoch_idx] = ninstances_base + epochs[epoch_idx]["train"]["num_entries"]

    # bundle arrays
    info = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'ninstances': ninstances
    }

    return info


def plot_training_history(info):

    # initialize output collection of figures
    figs = {}

    # make plot of training and validation loss vs epoch
    fig, ax = plt.subplots()
    ax.plot(info["train_loss"], color='dodgerblue', label='Training', linewidth=2)
    ax.plot(info["val_loss"], color='mediumorchid', label='Validation', linewidth=2)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlabel('Number of epochs', fontsize=14)
    ax.legend(fontsize=14)
    fig.tight_layout()
    figs['loss_vs_epoch'] = (fig, ax)

    # make plot of training and validation loss vs ninstances
    fig, ax = plt.subplots()
    permillion = info["ninstances"]/1e6
    ax.plot(permillion, info["train_loss"], color='dodgerblue', label='Training', linewidth=2)
    ax.plot(permillion, info["val_loss"], color='mediumorchid', label='Validation', linewidth=2)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlabel('Number of training instances (M)', fontsize=14)
    ax.legend(fontsize=14)
    fig.tight_layout()
    figs['loss_vs_ninstances'] = (fig, ax)

    return figs


if __name__=='__main__':

    # read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-o', '--outputdir', default='output_test')
    args = parser.parse_args()

    # load inputfile
    with open(args.inputfile, 'r') as f:
        info = json.load(f)

    # parse info and make plots
    info = parse_info(info)
    figs = plot_training_history(info)

    # make output directory
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # store plots
    for figtype, (fig, ax) in figs.items():
        outputfile = os.path.join(args.outputdir, f'{figtype}.png')
        fig.tight_layout()
        fig.savefig(outputfile)
        print(f'Saved figure {outputfile}.')
        ax.set_yscale('log')
        outputfile = os.path.join(args.outputdir, f'{figtype}_logy.png')
        fig.tight_layout()
        fig.savefig(outputfile)
        print(f'Saved figure {outputfile}.')
        ax.set_yscale('linear')
        ax.set_xscale('log')
        outputfile = os.path.join(args.outputdir, f'{figtype}_logx.png')
        fig.tight_layout()
        fig.savefig(outputfile)
        print(f'Saved figure {outputfile}.')
