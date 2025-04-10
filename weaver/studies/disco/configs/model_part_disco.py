import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../../../'))
from weaver.nn.model.ParticleTransformerDisco import ParticleTransformerDisco


class ParticleTransformerWrapper(torch.nn.Module):
    # see e.g. here:
    # https://github.com/jet-universe/particle_transformer/blob/main/
    # networks/example_ParticleTransformer.py

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = ParticleTransformerDisco(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask, decorrelate):
        # note: decorrelate is not used in forward pass
        return self.model(features, v=lorentz_vectors, mask=mask)

    def train_single_epoch(self, train_loader, dev, **kwargs):
        return self.model.train_single_epoch(train_loader, dev, **kwargs)


def get_model(data_config,
      disco_prediction_index=0, disco_alpha=0, disco_power=1,
      disco_label=None, disco_mass_min=None, disco_mass_max=None,
      **kwargs):
    
    # hard-coded settings
    # todo

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

    # make model arguments
    cfg = dict(
      input_dim = features_dims,
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
      # disco settings
      disco_prediction_index=disco_prediction_index,
      disco_alpha=disco_alpha,
      disco_power=disco_power,
      disco_label=disco_label,
      disco_mass_min=disco_mass_min,
      disco_mass_max=disco_mass_max
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
