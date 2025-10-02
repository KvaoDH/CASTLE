# castle/time_embedding.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import LearnableTimeScale, ConfScorer, TimeGapDecay

__all__ = ["TimeEmbeddingGRUEncoder"]

class TimeEmbeddingGRUEncoder(nn.Module):
    """
    Δ-aware GRU encoder (representation only):
      • Uses LearnableTimeScale τ to normalize deltas
      • Ages hidden state via TimeGapDecay (gap→forgetting)
      • Reuses DecayAware's conf_seq to weight writes (preferred)
      • Optional internal ConfScorer only if enabled

    Inputs
      x     : [B,T,D]   (finite tokens from DecayAware)
      mask  : [B,T,D] or None (1=observed, 0=missing). If None, inferred from x.
      delta : [B,T,D] or [B,T]  (5-min steps since previous sample)
      conf_override : [B,T,1] or [B,T] or None (from DecayAware.conf_seq). If None, see use_internal_conf.

    Returns
      y : [B, output_dim] if output_dim set, else [B, hidden_dim]
      H : [B,T,hidden_dim]
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int | None = None,
                 time_scale: float = 3600.0,
                 time_scale_module: LearnableTimeScale | None = None,
                 include_delta: bool = True,
                 time_embed_dim: int = 8,
                 pool_mode: str = "decay",   # {"decay","last","mean"}
                 dropout: float = 0.10,
                 use_internal_conf: bool = False,   # prefer DecayAware confidence
                 conf_min: float = 1e-3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.include_delta = bool(include_delta)
        self.time_embed_dim = int(time_embed_dim)
        self.pool_mode = str(pool_mode)
        self.use_internal_conf = bool(use_internal_conf)

        # Optional internal scorer (fallback only)
        self.conf_scorer = None
        if self.use_internal_conf:
            self.conf_scorer = ConfScorer(
                dim=self.input_dim, hidden=64, min_conf=conf_min, conf_from="x+mask", smooth_ema=None
            )
        self._last_conf = None  # [B,1]

        # Δ normalization + temporal parts
        self.ts_mod = time_scale_module if time_scale_module is not None else LearnableTimeScale(time_scale)
        self.time_embed = nn.Linear(1, self.time_embed_dim)   # small time-position embedding
        self.h_decay    = TimeGapDecay(1, hidden_dim)         # hidden decay from scalar Δ

        # GRU over augmented input (x ⊕ time_emb ⊕ [Δ]? as scalar)
        gru_in = self.input_dim + (1 if self.include_delta else 0) + self.time_embed_dim
        self.gru = nn.GRUCell(gru_in, hidden_dim)

        self.dropout = nn.Dropout(dropout) if (dropout and output_dim is not None) else nn.Identity()
        self.out = nn.Identity() if output_dim is None else nn.Sequential(
            nn.LayerNorm(hidden_dim),
            self.dropout,
            nn.Linear(hidden_dim, output_dim)
        )

    def _norm_delta_scalar(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Normalize Δ to d̄ = log1p(Δ_seconds / τ). Accepts [B,T,D] or [B,T]; returns [B,T,1].
        """
        if delta is None:
            raise ValueError("TimeEmbeddingGRUEncoder requires `delta`.")
        if delta.dim() == 3:
            d = delta.mean(dim=-1, keepdim=True)    # [B,T,1]
        elif delta.dim() == 2:
            d = delta.unsqueeze(-1)                  # [B,T,1]
        else:
            raise ValueError(f"delta must be [B,T] or [B,T,D], got {tuple(delta.shape)}")
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        tau = self.ts_mod().to(d.device).clamp_min(1e-6)
        d_seconds = d * 300.0  # 5-min steps → seconds
        return torch.log1p(d_seconds / tau)         # [B,T,1]

    def _pool(self, H: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "last":
            return H[:, -1, :]
        if self.pool_mode == "mean":
            return H.mean(dim=1)
        # decay-weighted pooling (default)
        B, T, Hdim = H.shape
        ages = torch.arange(T-1, -1, -1, device=H.device, dtype=H.dtype).view(1, T, 1)
        w = torch.exp(-0.05 * ages)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        return (H * w).sum(dim=1)

    @staticmethod
    def _coerce_conf(conf: torch.Tensor | None, B: int, T: int, device, dtype) -> torch.Tensor:
        """
        Accept [B,T,1] or [B,T] or None → return [B,T,1]; if None, return ones.
        """
        if conf is None:
            return torch.ones(B, T, 1, device=device, dtype=dtype)
        if conf.dim() == 2:
            conf = conf.unsqueeze(-1)           # [B,T,1]
        elif conf.dim() == 3 and conf.size(-1) == 1:
            pass
        else:
            raise ValueError(f"conf_override must be [B,T] or [B,T,1], got {tuple(conf.shape)}")
        return torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None,
                delta: torch.Tensor,
                conf_override: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, Dx = x.shape
        # finite for computation (DA already handled tokens; this is a safety net)
        x_f = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is None:
            mask = torch.isfinite(x).to(x.dtype)
        else:
            mask = torch.nan_to_num(mask, nan=0.0).clamp(0.0, 1.0)

        # Δ normalization → time embedding + hidden decay
        dbar = self._norm_delta_scalar(delta)       # [B,T,1]
        temb = torch.tanh(self.time_embed(dbar))    # [B,T,E]

        # choose confidence source
        if conf_override is not None:
            conf_all = self._coerce_conf(conf_override, B, T, x.device, x_f.dtype)  # [B,T,1]
            use_internal = False
        else:
            use_internal = self.use_internal_conf and (self.conf_scorer is not None)
            # placeholder; will fill per step if use_internal, else ones
            conf_all = torch.ones(B, T, 1, device=x.device, dtype=x_f.dtype)

        h = x_f.new_zeros(B, self.hidden_dim)
        H = []
        last_conf = None

        for t in range(T):
            # state aging by current Δ
            g_h = self.h_decay(dbar[:, t])          # [B,H]
            h = h * g_h

            x_t = x_f[:, t, :]
            m_t = mask[:, t, :]

            # per-step confidence
            if use_internal:
                conf_step = self.conf_scorer(x_t, m_t)    # [B,1]
                conf_all[:, t, :] = conf_step
            else:
                conf_step = conf_all[:, t, :]             # [B,1]
            last_conf = conf_step

            # GRU input
            if self.include_delta:
                inp = torch.cat([x_t, temb[:, t], dbar[:, t]], dim=-1)
            else:
                inp = torch.cat([x_t, temb[:, t]], dim=-1)

            h_new = self.gru(inp, h)

            # confidence-weighted write
            c = conf_step
            while c.dim() < h_new.dim():
                c = c.unsqueeze(-1)
            h = c * h_new + (1.0 - c) * h

            H.append(h.unsqueeze(1))

        H = torch.cat(H, dim=1)                     # [B,T,H]
        self._last_conf = last_conf                 # [B,1]
        y = self.out(self._pool(H))                 # [B,output] or [B,H]
        return y, H
