import os
import sys
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../../'))
from weaver.nn.model.ParticleNet import ParticleNet 


def get_model(data_config, **kwargs):
    
    # hard-coded settings
    conv_params = [
        (3, (16, 16, 16)),
        (3, (16, 16, 16)),
        (3, (16, 16, 16)),
    ]
    fc_params = [(16, 0.1)]
    use_fusion = True
    print(f'Found following convolutional architecture: {conv_params}')
    print(f'Found following fully connected architecture: {fc_params}')

    # settings defined in data config file
    jet_features_dims = len(data_config.input_dicts['jet_features'])
    num_classes = len(data_config.label_value)
    print(f'Found following input feature dims: {jet_features_dims}')
    print(f'Found following number of classes: {num_classes}')

    # get model
    model = ParticleNet(jet_features_dims, num_classes,
                        conv_params = conv_params,
                        fc_params = fc_params,
                        use_fusion = use_fusion,
                        use_fts_bn = kwargs.get('use_fts_bn', False),
                        use_counts = kwargs.get('use_counts', True),
                        for_inference = kwargs.get('for_inference', False)
    )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
    }

    print('Built following model:')
    print(model)
    print('Built following model info:')
    print(model_info)

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
