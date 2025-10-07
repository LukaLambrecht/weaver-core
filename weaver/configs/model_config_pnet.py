import os
import sys
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../'))
from weaver.nn.model.ParticleNet import ParticleNet


def get_model(data_config, **kwargs):
    
    # settings defined in data config file
    features_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    print(f'Found following input feature dims: {features_dims}')
    print(f'Found following number of classes: {num_classes}')

    # get model
    model = ParticleNet(features_dims, num_classes, **kwargs)

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
