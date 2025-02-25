import os
import sys
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../../'))
from weaver.nn.model.ParticleNetDisCo import ParticleNetDisCo


def get_model(data_config,
      disco_prediction_index=0, disco_alpha=0, disco_power=1,
      disco_label=None, disco_mass_min=None, disco_mass_max=None,
      **kwargs):
    
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
    features_dims = len(data_config.input_dicts['features'])
    num_classes = len(data_config.label_value)
    print(f'Found following input feature dims: {features_dims}')
    print(f'Found following number of classes: {num_classes}')

    # print disco settings
    print('  Found following DisCo parameters:')
    print(f'  disco_prediction_index: {disco_prediction_index}')
    print(f'  disco_alpha: {disco_alpha}')
    print(f'  disco_power: {disco_power}')
    print(f'  disco_label: {disco_label}')
    print(f'  disco_mass_min: {disco_mass_min}')
    print(f'  disco_mass_max: {disco_mass_max}')

    # get model
    model = ParticleNetDisCo(features_dims, num_classes,
                        conv_params = conv_params,
                        fc_params = fc_params,
                        use_fusion = use_fusion,
                        use_fts_bn = kwargs.get('use_fts_bn', False),
                        use_counts = kwargs.get('use_counts', True),
                        for_inference = kwargs.get('for_inference', False),
                        disco_prediction_index=disco_prediction_index,
                        disco_alpha=disco_alpha, disco_power=disco_power,
                        disco_label=disco_label,
                        disco_mass_min=disco_mass_min, disco_mass_max=disco_mass_max
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
