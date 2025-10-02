# castle/tabular.py
from __future__ import annotations
import torch
import torch.nn as nn

from .helper import ConfScorer

__all__ = ["TabularEncoder"]

class TabularEncoder(nn.Module):
    """
    Per-sample tabular encoder (representation only).
    Reuses DecayAware confidence when given; optional internal ConfScorer as fallback.

    Inputs:
      x             : [B,D] (finite tokens from DecayAware; e.g., T=1 → squeeze)
      mask          : [B,D] or None (1=observed, 0=missing) — used only if internal scorer is enabled
      conf_override : [B,1] or [B] or None (preferred; from DecayAware.conf_seq.squeeze(1))

    Returns:
      z : [B, output_dim]
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 use_mask: bool = False,
                 # confidence handling
                 use_internal_conf: bool = False,   # prefer DA confidence
                 conf_from: str = "x+mask",
                 min_conf: float = 1e-3,
                 dropout: float = 0.20):
        super().__init__()
        self.use_mask = bool(use_mask)
        self.input_dim = int(input_dim)
        self.use_internal_conf = bool(use_internal_conf)

        # Optional internal scorer (only if you enable it)
        self.conf_scorer = None
        if self.use_internal_conf:
            self.conf_scorer = ConfScorer(
                dim=self.input_dim, hidden=64, min_conf=min_conf, conf_from=conf_from, smooth_ema=None
            )

        in_dim = self.input_dim * (2 if self.use_mask else 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._last_conf = None  # [B,1]

    @staticmethod
    def _coerce_conf(conf: torch.Tensor | None, B: int, device, dtype) -> torch.Tensor:
        """
        Accept [B,1] or [B] or None → return [B,1]; if None, return ones.
        """
        if conf is None:
            return torch.ones(B, 1, device=device, dtype=dtype)
        if conf.dim() == 1:
            conf = conf.unsqueeze(-1)              # [B,1]
        elif conf.dim() == 2 and conf.size(-1) == 1:
            pass
        else:
            raise ValueError(f"conf_override must be [B] or [B,1], got {tuple(conf.shape)}")
        return torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                conf_override: torch.Tensor | None = None) -> torch.Tensor:
        # tiny safety (should already be finite from DA)
        x_carrier = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, D = x_carrier.shape

        # choose confidence source
        if conf_override is not None:
            conf = self._coerce_conf(conf_override, B, x_carrier.device, x_carrier.dtype)  # [B,1]
        elif self.use_internal_conf and (self.conf_scorer is not None):
            if mask is None:
                mask = torch.isfinite(x_carrier).to(x_carrier.dtype)
            else:
                mask = torch.nan_to_num(mask, nan=0.0).clamp(0.0, 1.0)
            conf = self.conf_scorer(x_carrier, mask)  # [B,1]
        else:
            conf = torch.ones(B, 1, device=x_carrier.device, dtype=x_carrier.dtype)  # neutral

        self._last_conf = conf  # stored for downstream routing if needed

        x_in = torch.cat([x_carrier, mask], dim=-1) if (self.use_mask and mask is not None) else x_carrier
        return self.net(x_in)
