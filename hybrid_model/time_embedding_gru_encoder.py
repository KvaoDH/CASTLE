import torch
import torch.nn as nn

class TimeEmbeddingGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_scale: float = 3600.0,
                 include_mask: bool = True, include_delta: bool = True):
        super().__init__()
        self.time_scale = time_scale
        self.include_mask = include_mask
        self.include_delta = include_delta

        self.time_embed = nn.Linear(1, 8)
        extra = 8 + (input_dim if include_mask else 0) + (1 if include_delta else 0)
        self.gru = nn.GRU(input_dim + extra, hidden_dim, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def _norm_delta(self, delta_t):
        if delta_t.dim() == 3:
            delta_t = delta_t.mean(dim=-1)
        delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.log1p(delta_t / self.time_scale)

    def forward(self, x, delta_t, mask=None):
        # x: [B, T, D]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        d_norm = self._norm_delta(delta_t)  # [B, T]
        time_feature = self.time_embed(d_norm.unsqueeze(-1))  # [B, T, 8]

        parts = [x, time_feature]
        if self.include_mask:
            if mask is None:
                mask = torch.isfinite(x).float()
            parts.append(mask)
        if self.include_delta:
            parts.append(d_norm.unsqueeze(-1))

        x_cat = torch.cat(parts, dim=-1)
        _, h = self.gru(x_cat)
        return self.out(torch.nan_to_num(h[-1], nan=0.0, posinf=0.0, neginf=0.0))
