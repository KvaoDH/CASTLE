# castle/decay_aware.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import LearnableTimeScale, ConfScorer, TimeGapDecay

__all__ = ["DecayAware"]

class DecayAware(nn.Module):
    """
    Missingness-only module:
      • Produces finite, mask-faithful tokens (xhat_seq)
      • Emits per-step imputation confidence (conf_seq)
      • Uses gap-aware drift toward a per-sequence fallback mean
    No sequence embedding, no hidden state, no pooling.

    Inputs:  x [B,T,D], x_mask [B,T] / [B,T,1] / [B,T,D], x_delta [B,T] / [B,T,1] / [B,T,D]
             x_mean [B,1,D] optional (fallback mean); if None, estimated from first k steps
    Returns: xhat_seq [B,T,D], conf_seq [B,T,1]
    """
    def __init__(
        self,
        input_size: int,
        *,
        device: str = "cpu",
        time_scale: float = 3600.0,
        time_scale_module: LearnableTimeScale | None = None,
        noise_scale: float = 0.0,
        conf_min: float = 1e-3,
        use_two_timescales: bool = True
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.device = device
        self.noise_scale = float(noise_scale)
        self.use_two_timescales = bool(use_two_timescales)

        # Δ normalization (shared τ)
        self.ts_mod = time_scale_module if time_scale_module is not None else LearnableTimeScale(time_scale)

        # Confidence scorer (no token modification)
        self.conf_scorer = ConfScorer(dim=self.input_size, hidden=64, min_conf=conf_min, conf_from="x+mask", smooth_ema=None)

        # Gap→decay (scalar Δ per step → per-feature decay)
        # We map [B,T,1] → [B,T,D]
        self.decay_fast = TimeGapDecay(in_dim=1, out_dim=self.input_size)
        self.decay_slow = TimeGapDecay(in_dim=1, out_dim=self.input_size) if self.use_two_timescales else None
        self.mix_gate_raw = nn.Parameter(torch.tensor(0.0))  # blend fast/slow; sigmoid→(0..1)

        # Optional: cached global mean (if you want to seed x_mean externally)
        self.register_buffer("x_global_mean", torch.zeros(1, 1, self.input_size))

        # Tiny input projection for internal stability (does NOT leak as representation)
        self.input_embed = nn.Identity()

        self._last_conf = None  # [B,1]

    @torch.no_grad()
    def set_global_mean(self, mean_vec: torch.Tensor):
        self.x_global_mean = mean_vec.view(1, 1, -1)

    def _norm_delta_scalar(self, d: torch.Tensor) -> torch.Tensor:
        """
        Normalize Δ to d̄ = log1p(Δ_seconds / τ); returns [B,T,1].
        Accepts [B,T], [B,T,1], or [B,T,D] and reduces to a scalar per step.
        """
        if d.dim() == 3 and d.size(-1) > 1:
            d = d.mean(dim=-1, keepdim=True)     # [B,T,1]
        elif d.dim() == 3 and d.size(-1) == 1:
            pass                                  # [B,T,1]
        elif d.dim() == 2:
            d = d.unsqueeze(-1)                   # [B,T,1]
        else:
            raise ValueError(f"x_delta must be [B,T], [B,T,1], or [B,T,D]; got {tuple(d.shape)}")

        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        tau = self.ts_mod().to(d.device).clamp_min(1e-6)
        d_seconds = d * 300.0  # 5-min steps → seconds
        return torch.log1p(d_seconds / tau)       # [B,T,1]

    def _blend_decay(self, dbar: torch.Tensor) -> torch.Tensor:
        """
        dbar: [B,T,1] → gamma_x: [B,T,D]
        """
        g_fast = self.decay_fast(dbar)
        if self.decay_slow is None:
            return g_fast
        g_slow = self.decay_slow(dbar)
        s = torch.sigmoid(self.mix_gate_raw).view(1, 1, 1)
        return s * g_fast + (1.0 - s) * g_slow

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, x_delta: torch.Tensor, x_mean: torch.Tensor | None = None):
        B, T, D = x.shape
        assert D == self.input_size, f"input_size mismatch: expected {self.input_size}, got {D}"

        # --- Broadcast/clean mask to [B,T,D] ---
        if x_mask.dim() == 2:                       # [B,T]
            x_mask = x_mask.unsqueeze(-1).expand(B, T, D)
        elif x_mask.dim() == 3 and x_mask.size(-1) == 1:  # [B,T,1]
            x_mask = x_mask.expand(B, T, D)
        m_full = torch.nan_to_num(x_mask, nan=0.0).clamp(0.0, 1.0)

        # --- Δ → normalized scalar per step ---
        dbar = self._norm_delta_scalar(x_delta)     # [B,T,1]
        gamma_x = self._blend_decay(dbar)           # [B,T,D]

        # --- Fallback mean ---
        if x_mean is None:
            k = min(6, T)
            x0 = torch.nan_to_num(x[:, :k, :], nan=0.0)
            m0 = m_full[:, :k, :]
            num = (x0 * m0).sum(dim=1)
            den = m0.sum(dim=1).clamp_min(1.0)
            x_mean = (num / den).unsqueeze(1)       # [B,1,D]

        # --- Iterate and impute (mask-faithful) ---
        x_finite = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        x_last = x_mean.squeeze(1).clone()          # start from mean

        XH, CONF = [], []
        last_conf = None

        for t in range(T):
            m_t = m_full[:, t, :]                   # [B,D]
            x_obs_t = torch.where(m_t > 0, x_finite[:, t, :], torch.zeros_like(x_finite[:, t, :]))

            # conservative imputation toward per-seq mean, gap-weighted
            x_fallback = gamma_x[:, t, :] * x_last + (1.0 - gamma_x[:, t, :]) * x_mean.squeeze(1)
            x_hat = m_t * x_obs_t + (1.0 - m_t) * x_fallback  # observed untouched

            # optional regularization noise ONLY on imputed entries
            if self.training and self.noise_scale > 0.0:
                noise = torch.randn_like(x_hat) * self.noise_scale
                x_hat = x_hat + (1.0 - m_t) * noise

            # per-step imputation confidence (no token injection)
            conf_t = self.conf_scorer(x_hat, m_t)   # [B,1]
            last_conf = conf_t

            # update "last observed" tracker only where we had observations
            x_last = torch.where(m_t.bool(), x_obs_t, x_last)

            # collect
            XH.append(self.input_embed(x_hat).unsqueeze(1))  # keep identity; safe hook
            CONF.append(conf_t.unsqueeze(1))

        xhat_seq = torch.cat(XH, dim=1)             # [B,T,D]
        conf_seq = self.conf_scorer(xhat_seq, m_full)           # [B,T,1]
        self._last_conf = conf_seq[:, -1, :]                 # [B,1]
        return xhat_seq, conf_seq
