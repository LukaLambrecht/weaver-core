# Implementation of uniformity or flatness loss between two variables

import os
import sys
import numpy as np
import torch

# local re-implementation of torch.histogram, to enable CUDA support
sys.path.append(os.path.dirname(__file__))
from histogram import histogram as torch_histogram


def flatnessloss(var_1, var_2, var_1_bins, weight=None):
    # calculate flatness loss between var_1 and var_2.
    # more info and references here:
    # https://github.com/LukaLambrecht/mass-decorrelation-tutorials/blob/main/python/uGBFL.py
    # input arguments:
    # - var_1: first variable to decorrelate (e.g. classifier score).
    # - var_2: second variable to decorrelate (e.g. mass).
    # - var_1_bins: binning for the first variable used for computing
    #   the groups of the second variable whose CDFs will be compared.
    # - weight: per-instance weight.
    #   note: sum of weights should add up to the number of instances.
    #   default: weight 1 for each instance.
    # notes:
    # - var_1, var_2 and weight should all be 1D tensors
    #   with the same number of entries.

    # make bins and cumulative distribution for var_2
    var_2_clipped = torch.clip(var_2, min=torch.quantile(var_2, 0.05), max=torch.quantile(var_2, 0.95))
    var_2_bins = torch.linspace(torch.amin(var_2_clipped), torch.amax(var_2_clipped), 10)
    var_2_hist = torch_histogram(var_2_clipped, edges=var_2_bins, weights=weight)
    var_2_cdf = torch.cumsum( var_2_hist, 0 )
    norm = var_2_cdf[-1]
    if norm > 0: var_2_cdf = torch.div(var_2_cdf, norm)
    # initializations
    loss = 0.
    var_1_bin_weights = torch.zeros(len(var_1_bins)-1)
    # loop over var_1 bins
    for bin_idx in range(len(var_1_bins)-1):
        # select var_2 in given var_1 bin
        var_1_low = var_1_bins[bin_idx]
        var_1_high = var_1_bins[bin_idx+1]
        var_2_in_bin = var_2_clipped[torch.where( (var_1>var_1_low) & (var_1<var_1_high) )]
        weights_in_bin = None
        if weight is not None: weights_in_bin = weight[torch.where( (var_1>var_1_low) & (var_1<var_1_high) )]
        # make cumulative distribution for var_2 in this bin
        hist = torch_histogram(var_2_in_bin, edges=var_2_bins, weights=weights_in_bin)
        cdf = torch.cumsum( hist, 0 )
        norm = cdf[-1]
        if norm > 0: cdf = torch.div(cdf, norm)
        # calculate difference w.r.t. total cumulative distribution
        sqdiff = torch.sum(torch.square( cdf - var_2_cdf ))
        #var_1_bin_weights[bin_idx] = len(var_2_in_bin)/len(var_2)
        var_1_bin_weights[bin_idx] = 1
        loss += var_1_bin_weights[bin_idx] * sqdiff
    return loss
