import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../../'))
from weaver.nn.model.ParticleTransformer import ParticleTransformer


class ParticleTransformerWrapper(torch.nn.Module):
    # see e.g. here:
    # https://github.com/jet-universe/particle_transformer/blob/main/
    # networks/example_ParticleTransformer.py

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.model(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    
    # hard-coded settings
    # todo

    # settings defined in data config file
    jet_features_dims = len(data_config.input_dicts['jet_features'])
    num_classes = len(data_config.label_value)
    print(f'Found following input feature dims: {jet_features_dims}')
    print(f'Found following number of classes: {num_classes}')

    # make model arguments
    cfg = dict(
      input_dim = jet_features_dims,
      num_classes = num_classes,
      # network configurations
      pair_input_dim=4,
      pair_extra_dim=0,
      remove_self_pair=False,
      use_pre_activation_pair=True,
      embed_dims=[6, 12, 6],
      pair_embed_dims=[6, 6, 6],
      num_heads=2,
      num_layers=4,
      num_cls_layers=2,
      block_params=None,
      cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
      fc_params=[],
      activation='gelu',
      # misc
      trim=True,
      for_inference=False,
      use_amp=False,
    )
    cfg.update(**kwargs)

    print('Model config:')
    print(json.dumps(cfg, indent=2))

    model = ParticleTransformerWrapper(**cfg)

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
