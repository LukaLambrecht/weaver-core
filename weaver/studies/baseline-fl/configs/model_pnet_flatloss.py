import os
import sys
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../../'))
from weaver.nn.model.ParticleNetFL import ParticleNetFL


def get_model(data_config,
      flatloss_prediction_index=0,
      flatloss_bins='auto',
      flatloss_alpha=0,
      flatloss_label=None,
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

    # print flatloss settings
    print('  Found following DisCo parameters:')
    print(f'  flatloss_prediction_index: {flatloss_prediction_index}')
    print(f'  flatloss_bins: {flatloss_bins}')
    print(f'  flatloss_alpha: {flatloss_alpha}')
    print(f'  flatloss_label: {flatloss_label}')

    # get model
    model = ParticleNetFL(features_dims, num_classes,
                        conv_params = conv_params,
                        fc_params = fc_params,
                        use_fusion = use_fusion,
                        use_fts_bn = kwargs.get('use_fts_bn', False),
                        use_counts = kwargs.get('use_counts', True),
                        for_inference = kwargs.get('for_inference', False),
                        flatloss_prediction_index=flatloss_prediction_index,
                        flatloss_bins=flatloss_bins,
                        flatloss_alpha=flatloss_alpha,
                        flatloss_label=flatloss_label
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
