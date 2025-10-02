# castle/stream_cond.py
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["StreamCondBuilder"]

class StreamCondBuilder(nn.Module):
    """
    Builds a small per-stream condition vector (Cps=5) for fusion.
      cond = [ conf_mean, missing_rate, recent_obs_rate(last_k),
               mean_log1p_delta_over_tau, last_log1p_delta_over_tau ]  # [B,5]

    Inputs already sanitized by DecayAware — no NaN handling here.
    """
    def __init__(self, time_scale_module):
        super().__init__()
        self.ts_mod = time_scale_module

    def _norm_delta(self, delta: torch.Tensor) -> torch.Tensor:
        # delta -> [B,T,1] of log1p(Δ_seconds / τ)
        if delta.dim() == 2:          # [B,T] -> [B,T,1]
            d = delta.unsqueeze(-1)
        elif delta.dim() == 3:        # [B,T,D] -> [B,T,1]
            d = delta.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"delta must be [B,T] or [B,T,D], got {tuple(delta.shape)}")

        tau = self.ts_mod().to(d.device).clamp_min(1e-6)
        d_seconds = d.clamp_min(0) * 300.0          # 5-min steps
        return torch.log1p(d_seconds / tau)         # [B,T,1]

    def forward(self, x, mask, delta, conf, last_k: int) -> torch.Tensor:
        B, T, D = x.shape

        # conf_mean
        if conf.dim() == 2: conf = conf.unsqueeze(-1)      # [B,T,1]
        conf_mean = conf.mean(dim=1)                       # [B,1]

        # missing_rate, recent_obs over last_k
        miss_rate = (1.0 - mask.mean(dim=(1,2), keepdim=False)).unsqueeze(-1)  # [B,1]
        k = max(1, min(int(last_k), T))
        recent_obs = mask[:, -k:, :].mean(dim=(1,2), keepdim=False).unsqueeze(-1)  # [B,1]

        # Δ stats (log1p normalized)
        dbar = self._norm_delta(delta)                     # [B,T,1]
        d_mean = dbar.mean(dim=1)                          # [B,1]
        d_last = dbar[:, -1, :]                            # [B,1]

        return torch.cat([conf_mean, miss_rate, recent_obs, d_mean, d_last], dim=1)  # [B,5]
