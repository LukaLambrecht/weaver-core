#######################################################################
# Simple shallow feedforward neural network with DisCo implementation #
#######################################################################

# Mainly for testing purposes before attempting DisCo on more complicated models.


import os
import sys
import json
import tqdm
import torch
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(weavercoredir)
from weaver.utils.disco import distance_correlation
from weaver.utils.nn.tools import _flatten_preds


class DiscoNetwork(torch.nn.Module):
    
    def __init__(self, n_inputs,
                 architecture=[8],
                 num_classes=2,
                 disco_prediction_index=0,
                 disco_alpha=0,
                 disco_power=1,
                 disco_label=None,
                 disco_mass_min=None,
                 disco_mass_max=None,
                 **kwargs):
        super().__init__(**kwargs)

        # initializations
        self.n_inputs = n_inputs
        self.architecture = architecture
        self.num_classes = num_classes
        self.disco_prediction_index = disco_prediction_index
        self.disco_alpha = disco_alpha
        self.disco_power = disco_power
        self.disco_label = disco_label
        self.disco_mass_min = disco_mass_min
        self.disco_mass_max = disco_mass_max

        # define intermediate layers
        n_units = [n_inputs] + architecture
        layers = []
        for idx in range(len(n_units)-1):
            n_in = n_units[idx]
            n_out = n_units[idx+1]
            layers.append( torch.nn.BatchNorm1d(n_in) )
            layers.append( torch.nn.Linear(n_in, n_out) )
            layers.append( torch.nn.LeakyReLU(negative_slope=0.5) )
        
        # define output layer
        layers.append( torch.nn.Linear(architecture[-1], num_classes) )
        layers.append( torch.nn.Softmax(dim=1) )
        self.model = torch.nn.Sequential(*layers)

        # loss function 
        self.celossfunction = torch.nn.CrossEntropyLoss()
        
    def forward(self, X, _):
        # do forward pass
        # note: the _ argument is needed for syntax,
        #       since the inputs (as specified in the data config file)
        #       consist of 'input_features' and 'decorrelate',
        #       but the second one is not used in the forward pass,
        #       only in the loss function.

        # first flatten the input data starting at dimension 1,
        # so that shape (batch size, .., .., etc) is transormed
        # into (batch size, flattened input size).
        X = X.flatten(start_dim=1)
        # forward pass
        output = self.model(X)
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
        totalloss = celoss + self.disco_alpha * discoloss
        return (totalloss, {'BCE': celoss.item(), 'DisCo': discoloss.item()})

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
        self.model.train()

        # initializations
        loss_value = 0
        batch_idx = 0
        data_config = train_loader.dataset.config

        # loop over batches
        with tqdm.tqdm(train_loader) as tq:
            for X, y, _ in tq:

                # prepare input data and labels
                # note: data_config.input_names is a list of names defined
                #       in the 'inputs' section of the data config file.
                #       Here, we take the 'input_features' key for the input,
                #       and the "decorrelate" key for the mass!
                # note: data_config.label_names is a tuple holding only one element,
                #       called '_labels_', corresponding to the combined label.
                if not 'input_features' in data_config.input_names:
                    msg = 'Inputs specified in data config must contain the key'
                    msg += ' "input_features" for the input variables.'
                    raise Exception(msg)
                inputs = [X['input_features'].to(dev)]
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
                predictions = self.forward(*inputs, None)

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
                loss = loss.item()
                tq.set_postfix({
                  'Loss': '{:.5f}'.format(loss),
                  'BCE': '{:.3f}'.format(lossvalues['BCE']),
                  'DisCo': '{:.3f}'.format(lossvalues['DisCo'])
                })

                # update counters
                batch_idx += 1

                # break after a given number of batches
                if 'steps_per_epoch' in kwargs:
                    steps_per_epoch = kwargs['steps_per_epoch']
                    if batch_idx >= steps_per_epoch: break

        # update scheduler (if it was not done per batch)
        if scheduler and not getattr(scheduler, '_update_per_step', False): scheduler.step()


def get_model(data_config, architecture=[8], 
      disco_prediction_index=0, disco_alpha=0, disco_power=1,
      disco_label=None, disco_mass_min=None, disco_mass_max=None,
      **kwargs):
    # settings
    print('Info from get_model:')
    print('  Will instantiate a DiscoNetwork model.')
    _, n_features, feature_length = data_config.input_shapes['input_features']
    print(f'  Found {n_features} features each with length {feature_length}.')
    n_inputs = n_features * feature_length
    print(f'  Flattened input dimension: {n_inputs}.')
    num_classes = len(data_config.label_value)
    print(f'  Found {num_classes} output classes.')
    if isinstance(architecture, str):
        architecture = json.loads(architecture)
    print(f'  Found following architecture: {architecture}')
    print('  Found following DisCo parameters:')
    print(f'  disco_prediction_index: {disco_prediction_index}')
    print(f'  disco_alpha: {disco_alpha}')
    print(f'  disco_power: {disco_power}')
    print(f'  disco_label: {disco_label}')
    print(f'  disco_mass_min: {disco_mass_min}')
    print(f'  disco_mass_max: {disco_mass_max}')
    model = DiscoNetwork(n_inputs, architecture=architecture,
              disco_prediction_index=disco_prediction_index,
              disco_alpha=disco_alpha, disco_power=disco_power,
              disco_label=disco_label,
              disco_mass_min=disco_mass_min, disco_mass_max=disco_mass_max)
    # model info
    model_info = {
      'input_names': list(data_config.input_names),
      'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
      'output_names': ['softmax'],
      'dynamic_axes': {**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}}
    }
    return model, model_info
