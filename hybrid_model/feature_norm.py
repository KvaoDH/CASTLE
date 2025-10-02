import torch
import torch.nn as nn

class FeatureNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
