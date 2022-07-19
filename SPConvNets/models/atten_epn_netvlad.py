"""
network architechture for place recognition (Oxford dataset) with attention
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import vgtk
import SPConvNets.utils as M
import SPConvNets.models.pr_so3net_pn as frontend
import config as cfg


class Atten_EPN_NetVLAD(nn.Module):
    def __init__(self, opt):
        super(Atten_EPN_NetVLAD, self).__init__()
        self.opt = opt

        self.atten = torch.nn.MultiheadAttention(3, 3, batch_first=True)

        # epn param
        # mlps=[[64], [128]]
        # out_mlps=[128, cfg.LOCAL_FEATURE_DIM]
        self.mlps=[[32], [64]]
        out_mlps=[self.mlps[-1][0], cfg.LOCAL_FEATURE_DIM]
        
        self.epn = frontend.build_model(self.opt, self.mlps, out_mlps, outblock='linear')

        self.netvlad = M.NetVLADLoupe(feature_size=cfg.LOCAL_FEATURE_DIM, max_samples=cfg.NUM_SELECTED_POINTS, cluster_size=64,
                                     output_dim=cfg.GLOBAL_DESCRIPTOR_DIM, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        '''
        INPUT: B, N, D
        Local Feature: B, cfg.NUM_SELECTED_POINTS, cfg.LOCAL_FEATURE_DIM
        Global Feature: B, cfg.GLOBAL_DESCRIPTOR_DIM
        '''        
        # Use attention to choose points and downsample
        with torch.no_grad():
            x_atten, atten_weight = self.atten(x, x, x)
        atten_weight_sum = torch.sum(atten_weight, 2) # B, np
        
        # initialize local feature
        x_frontend = torch.zeros((x.shape[0], cfg.NUM_SELECTED_POINTS, cfg.LOCAL_FEATURE_DIM), device=x.device)
        x_equivariant = torch.zeros((x.shape[0], self.mlps[-1][0], cfg.NUM_SELECTED_POINTS, self.opt.model.kanchor), device=x.device)
        x_downsampled = torch.zeros((x.shape[0], cfg.NUM_SELECTED_POINTS, 3), device=x.device)

        if x.shape[0] >= 1:
            for i in range(x.shape[0]):
                current_attn = atten_weight_sum[i, :]
                current_pcd = x[i, torch.topk(current_attn, cfg.NUM_SELECTED_POINTS)[1].squeeze(), :].unsqueeze(0)
                x_downsampled[i, :, :] = current_pcd
                x_frontend[i, :, :], x_equivariant[i, :, :] = self.epn(current_pcd)
        else:
            print('x.shape[0]', x.shape[0])

        x = self.netvlad(x_frontend)

        return x, x_equivariant, # x_downsampled, x_frontend