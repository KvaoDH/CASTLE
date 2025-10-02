import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_mask: bool = True):
        super().__init__()
        self.use_mask = use_mask
        in_dim = input_dim * (2 if use_mask else 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, mask=None):
        # x may contain NaNs; build mask if not provided
        if mask is None:
            mask = torch.isfinite(x).float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_in = torch.cat([x, mask], dim=-1) if self.use_mask else x
        return self.net(x_in)
