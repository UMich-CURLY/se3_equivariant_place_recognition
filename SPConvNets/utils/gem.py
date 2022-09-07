import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        x_input: B, N, D_local
        x_output: B, D_global
        """
        # This implicitly applies ReLU on x (clamps negative values)
        x_out = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(1), 4)).pow(1./self.p)
        return x_out