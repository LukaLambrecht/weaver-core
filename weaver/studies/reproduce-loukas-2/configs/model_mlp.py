#############################################
# Simple shallow feedforward neural network #
#############################################

# Mainly for syntax testing purposes.
# See https://cms-ml.github.io/documentation/inference/particlenet.html


import os
import sys
import torch


class Network(torch.nn.Module):

    def __init__(self, n_inputs,
        num_classes=2, architecture=[8], **kwargs):
        super().__init__(**kwargs)

        # initializations
        self.n_inputs = n_inputs
        self.num_classes = num_classes
        self.architecture = architecture

        # define hidden layers
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

    def forward(self, x):
        # first flatten the input data starting at dimension 1,
        # so that shape (batch size, .., .., etc) is transormed
        # into (batch size, flattened input size).
        x = x.flatten(start_dim=1)
        # forward pass
        output = self.model(x)
        return output


def get_model(data_config, architecture=[4,4], **kwargs):
    # settings
    _, n_features, feature_length = data_config.input_shapes['input_features']
    print(f'Found {n_features} features each with length {feature_length}.')
    n_inputs = n_features * feature_length
    print(f'Flattened input dimension: {n_inputs}.')
    num_classes = len(data_config.label_value)
    print(f'Found {num_classes} output classes.')
    print(f'Found following architecture: {architecture}.')
    # get the model
    model = Network(n_inputs, num_classes=num_classes, architecture=architecture)
    # model info
    model_info = {
      'input_names': list(data_config.input_names),
      'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
      'output_names': ['softmax'],
      'dynamic_axes': {**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}}
    }
    print('Built following model and model info:')
    print(model)
    print(model_info)
    return model, model_info

def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
