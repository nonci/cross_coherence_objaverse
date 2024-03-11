"""
The MIT License (MIT)
Originally created sometime in 2019.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab

ADDED SOME DOCUMENTATION.
"""

import torch
from torch import nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, in_feat_dims, out_channels, use_b_norm, dropout,
                 non_linearity=nn.ReLU(inplace=True), closure=None):
        ''' Takes:
		in_feat_dims: input dimensions of 1st layer
		out_channels: list of channel dimensions, one for each desired layer
		... '''
        super(MLPDecoder, self).__init__()

        previous_feat_dim = in_feat_dims
        all_ops = []

        for depth in range(len(out_channels)):
            out_dim = out_channels[depth]
            
            if depth == len(out_channels) - 2:   # before middle layer
                if dropout:
                    all_ops.append(nn.Dropout(0.2 if dropout==True else dropout))
                    
            affine_op = nn.Linear(previous_feat_dim, out_dim, bias=True)
            all_ops.append(affine_op)

            if depth < len(out_channels) - 1:   # after first layer and after middle layer
                if use_b_norm:
                    all_ops.append(nn.BatchNorm1d(out_dim))

                if non_linearity is not None:
                    all_ops.append(non_linearity)
            
            previous_feat_dim = out_dim

        if closure is not None:
            all_ops.append(closure)

        self.net = nn.Sequential(*all_ops)

    def forward(self, x):
        return self.net(x)
