#######################################################################
# Simple shallow feedforward neural network with DisCo implementation #
#######################################################################

# Mainly for testing purposes before attempting DisCo on more complicated models.


import os
import sys
import tqdm
import torch

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)
from python.disco import distance_correlation

weavercoredir = os.path.abspath(os.path.join(thisdir, '../..'))
sys.path.append(weavercoredir)
from weaver.utils.nn.tools import _flatten_preds


class DiscoNetwork(torch.nn.Module):
    
    def __init__(self, n_inputs,
                 architecture=[8],
                 num_classes=2,
                 disco_alpha=0,
                 disco_power=1):
        super().__init__()

        if num_classes != 2:
            raise NotImplementedError()

        # initializations
        self.architecture = architecture
        self.num_classes = num_classes
        self.disco_alpha = disco_alpha
        self.disco_power = disco_power

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

        # loss function and optimizer 
        self.bcelossfunction = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
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
        # loss = binary cross-entropy + alpha * distance correlation
        bceloss = self.bcelossfunction(predictions, labels)
        discoloss = distance_correlation(predictions, mass, power=self.disco_power)
        totalloss = bceloss + self.disco_alpha * discoloss
        return (totalloss, {'BCE': bceloss.item(), 'DisCo': discoloss.item()})

    def train_single_epoch(self, train_loader, dev, **kwargs):
        # custom training loop (for one epoch).
    
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
                labels = y[data_config.label_names[0]].float().to(dev)

                # reset optimizer
                self.optimizer.zero_grad()

                # forward pass
                predictions = self.forward(*inputs, None)

                # take only first dimension in the prediction,
                # assuming that is for signal
                # (todo: find cleaner way)
                predictions = predictions[:, 0]

                # calculate loss
                loss, lossvalues = self.loss(predictions, labels, mass)

                # backpropagation
                loss.backward()
                self.optimizer.step()

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


def get_model(data_config, **kwargs):
    # settings
    _, n_features, feature_length = data_config.input_shapes['input_features']
    print(f'Found {n_features} features each with length {feature_length}.')
    n_inputs = n_features * feature_length
    print(f'Flattened input dimension: {n_inputs}.')
    num_classes = len(data_config.label_value)
    print(f'Found {num_classes} output classes.')
    architecture = [4, 4]
    disco_alpha = 0
    disco_power = 1
    model = DiscoNetwork(n_inputs, architecture=architecture,
              disco_alpha=disco_alpha, disco_power=disco_power)
    # model info
    model_info = {
      'input_names': list(data_config.input_names),
      'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
      'output_names': ['softmax'],
      'dynamic_axes': {**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}}
    }
    return model, model_info
