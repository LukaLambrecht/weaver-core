import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(weavercoredir)
from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger


class ParticleTransformerTaggerWrapper(torch.nn.Module):
    # see e.g. here:
    # https://github.com/jet-universe/particle_transformer/blob/main/
    # networks/example_ParticleTransformer.py

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = ParticleTransformerTagger(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, pf_points, pf_features, pf_vectors, pf_mask, sv_points, sv_features, sv_vectors, sv_mask):
        return self.model(pf_features, pf_v=pf_vectors, pf_mask=pf_mask,
                          sv_x=sv_features, sv_v=sv_vectors, sv_mask=sv_mask)


def get_model(data_config, **kwargs):
    
    # settings defined in data config file
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    pf_points_dims = len(data_config.input_dicts['pf_points'])
    sv_points_dims = len(data_config.input_dicts['sv_points'])
    num_classes = len(data_config.label_value)
    print(f'Found following particle input feature dims: {pf_features_dims}')
    print(f'Found following secondary vertex input feature dims: {sv_features_dims}')
    print(f'Found following particle point dims: {pf_points_dims}')
    print(f'Found following secondary vertex point dims: {sv_points_dims}')
    print(f'Found following number of classes: {num_classes}')

    # set arguments
    # make model arguments
    cfg = dict(
      # basic
      pf_input_dim = pf_features_dims,
      sv_input_dim = sv_features_dims,
      num_classes = num_classes,
      # network configurations
      pair_input_dim=4,
      pair_extra_dim=0,
      remove_self_pair=False,
      use_pre_activation_pair=True,
      embed_dims=[128, 256, 128],
      pair_embed_dims=[32, 64, 32],
      num_heads=8,
      num_layers=6,
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

    # get model
    model = ParticleTransformerTaggerWrapper(**cfg)

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
