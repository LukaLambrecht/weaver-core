'''
ParticleNet with DisCo

Originally copied from nn/model/ParticleNet.py in this repository.
Modified to include DisCo loss.
'''

import os
import sys
import math
import json
import tqdm
import torch
import numpy as np
import torch.nn as nn

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(weavercoredir)
from weaver.utils.disco import distance_correlation
from weaver.utils.nn.tools import _flatten_preds


def knn(x, k):
    '''
    Return k nearest neighbours
    Input arguments:
    - x: tensor of shape (batch size, number of coordinates, number of points)
    - k: number of nearest neighbours to return
    Returns:
    - tensor of shape (batch size, number of points, k) with indices
      of k nearest points to the given point in the given instance.
    '''
    # calculate matrix of distances (?) between all points
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # (batch size, npoints, npoints) (?)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) # (batch size, npoints, npoints) (?)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # (batch size, npoints, npoints) (?)
    # check if k is not too large compared to the number of points
    if k >= x.shape[2]:
        msg = f'Error in knn calculation: the {k} nearest neighbours were requested,'
        msg += f' but only {x.shape[2]} points are present per instance.'
        raise Exception(msg)
    # get indices of k largest elements
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:] # (batch size, npoints, k)
    return idx


def get_graph_feature_v1(x, k, idx):
    '''
    (?)
    Note: v1 is faster on GPU
    Input arguments:
    - x: tensor of shape (batch size, number of coordinates, number of points).
    - k: integer number of nearest neighbours to use.
    - idx: tensor with indices of nearest neighbours, of shape (batch size, number of points, k).
      (output of knn function, see above).
    '''
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)
    # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)
    # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()
    # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)
    # -> (batch_size, 2*num_dims, num_points, k)
    return fts


def get_graph_feature_v2(x, k, idx):
    '''
    (?)
    Note: v2 is faster on CPU
    Input arguments: same as v1
    '''
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)
    # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)
    # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()
    # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)
    # -> (batch_size, 2*num_dims, num_points, k)
    return fts


class EdgeConvBlock(nn.Module):
    '''
    EdgeConv layer.
    See https://arxiv.org/pdf/1801.07829
    Can be described as follows:
        x_i^(l+1) = max_{j in N(i)} ReLU(
         theta * (x_j^(l) - x_i^(l)) + phi * x_i^(l) )
    where:
      - N(i) is the set of neighbors of i
      - x_i^(l) is the i'th point in the given instance in layer l (?)
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    '''

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn: x = bn(x)
            if act: x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNetDisCo(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=False,
                 disco_prediction_index=0,
                 disco_alpha=0,
                 disco_power=1,
                 disco_label=None,
                 disco_mass_min=None,
                 disco_mass_max=None,
                 **kwargs):
        # note: kwargs are passed to nn.Module constructor
        super(ParticleNetDisCo, self).__init__(**kwargs)

        # set batch norm for features
        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn: self.bn_fts = nn.BatchNorm1d(input_dims)

        # (what is this?)
        self.use_counts = use_counts

        # set architecture of edge convolution layers
        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        # set fusion (what is this?)
        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        # (what is this?)
        self.for_segmentation = for_segmentation

        # set architecture for fully connected layers
        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        # (what is this?)
        self.for_inference = for_inference

        # initializations for DisCo method
        self.disco_prediction_index = disco_prediction_index
        self.disco_alpha = disco_alpha
        self.disco_power = disco_power
        self.disco_label = disco_label
        self.disco_mass_min = disco_mass_min
        self.disco_mass_max = disco_mass_max

        # initialize cross-entropy loss
        self.celossfunction = torch.nn.CrossEntropyLoss()


    def forward(self, points, features, mask, _):
        # do forward pass
        # note: the _ argument is needed for syntax,
        #       since the inputs (as specified in the data config file)
        #       consist of vectors, features, mask, and decorrelate;
        #       the latter is automatically passed to the forward pass,
        #       but it is not used in the forward pass,
        #       only in the loss function.
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask
        
        if self.for_segmentation: x = fts
        else:
            if self.use_counts: x = fts.sum(dim=-1) / counts
            else: x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output


    def loss(self, predictions, labels, mass):
        # custom loss function.
        # loss = cross-entropy + alpha * distance correlation

        # calculate cross-entropy loss
        predictions_flat, labels_flat, _ = _flatten_preds(predictions, label=labels)
        celoss = self.celossfunction(predictions_flat, labels_flat)

        # prepare data for calculation of distance correlation
        # note: can only take one dimension in the prediction!
        predictions_column = predictions[:, self.disco_prediction_index]

        # optional: do preselection on which predictions/mass values to use in the disco calculation
        mask = np.ones(len(predictions_column))
        if self.disco_label is not None: mask = np.where(labels != self.disco_label, 0, mask)
        if self.disco_mass_min is not None: mask = np.where(mass < self.disco_mass_min, 0, mask)
        if self.disco_mass_max is not None: mask = np.where(mass > self.disco_mass_max, 0, mask)
        mask = mask.astype(bool)
        predictions_column = predictions_column[mask]
        mass = mass[mask]

        # calculate distance correlation
        discoloss = distance_correlation(predictions_column, mass, power=self.disco_power)

        # make the sum
        if( math.isnan(discoloss.item()) or np.isnan(discoloss.item()) ):
            totalloss = (1 + self.disco_alpha) * celoss
        else: 
            totalloss = celoss + self.disco_alpha * discoloss
        return (totalloss, {'total': totalloss.item(), 'BCE': celoss.item(), 'DisCo': discoloss.item()})


    def train_single_epoch(self, train_loader, dev, **kwargs):
        # custom training loop (for one epoch).

        # get optimizer and scheduler from keyword arguments
        # (optional according to the syntax, but in practice always provided)
        optimizer = kwargs['optimizer']
        scheduler = kwargs['scheduler']

        # get optional keyword arguments
        grad_scaler = None
        if 'grad_scaler' in kwargs.keys(): grad_scaler = kwargs['grad_scaler']

        # set model ready for training
        self.train()

        # initializations
        batch_idx = 0
        data_config = train_loader.dataset.config
        history = []

        # loop over batches
        with tqdm.tqdm(train_loader) as tq:
            for X, y, _ in tq:

                # prepare input data and labels
                # note: data_config.input_names is a list of names defined
                #       in the 'inputs' section of the data config file.
                # note: data_config.label_names is a tuple holding only one element,
                #       called '_labels_', corresponding to the combined label.
                if not 'points' in data_config.input_names:
                    msg = 'Inputs specified in data config must contain the key'
                    msg += ' "points" for the input point coordinates.'
                    raise Exception(msg)
                points = X['points'].to(dev)
                if not 'features' in data_config.input_names:
                    msg = 'Inputs specified in data config must contain the key'
                    msg += ' "features" for the input point features.'
                    raise Exception(msg)
                features = X['features'].to(dev)
                if not 'mask' in data_config.input_names:
                    msg = 'Inputs specified in data config must contain the key'
                    msg += ' "mask" for the input point coordinates.'
                    raise Exception(msg)
                mask = X['mask'].to(dev)
                if not 'decorrelate' in data_config.input_names:
                    msg = 'Inputs specified in data config must contain the key'
                    msg += ' "decorrelate" for the decorrelation variable.'
                    raise Exception(msg)
                mass = X['decorrelate'].to(dev).squeeze()
                labels = y[data_config.label_names[0]].long().to(dev)
                # (note: conversion to long seems to be needed for CrossEntropyLoss,
                #  while for BCELoss, conversion to float seems to be required)

                # reset optimizer
                optimizer.zero_grad()

                # forward pass
                predictions = self.forward(points, features, mask, None)

                # calculate loss
                loss, lossvalues = self.loss(predictions, labels, mass)

                # backpropagation
                if grad_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                if scheduler and getattr(scheduler, '_update_per_step', False):
                    scheduler.step()

                # print total loss
                tq.set_postfix({
                  'Loss': '{:.5f}'.format(lossvalues['total']),
                  'BCE': '{:.3f}'.format(lossvalues['BCE']),
                  'DisCo': '{:.3f}'.format(lossvalues['DisCo'])
                })

                # append to history
                history.append(lossvalues)

                # update counters
                batch_idx += 1

                # break after a given number of batches
                if 'steps_per_epoch' in kwargs:
                    steps_per_epoch = kwargs['steps_per_epoch']
                    if batch_idx >= steps_per_epoch: break

        # update scheduler (if it was not done per batch)
        if scheduler and not getattr(scheduler, '_update_per_step', False): scheduler.step()

        # parse history from list of dicts to dict of lists
        history_new = {}
        for key in history[0].keys():
            history_new[key] = [el[key] for el in history]
        history = history_new

        # return history
        return history
