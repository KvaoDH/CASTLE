import torch
import torch.nn as nn
import torch.nn.functional as F

class MonotoneDecay(nn.Module):
    """
    Monotone decay mapping: gamma = exp(-softplus(W * d + b)) in (0, 1].
    Ensures gamma decreases as d increases (robust under site shift).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, delta_norm):
        # delta_norm: [B, D_in]
        return torch.exp(-F.softplus(self.lin(delta_norm)))
