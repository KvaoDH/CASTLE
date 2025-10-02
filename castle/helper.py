# castle/helper.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnableTimeScale",
    "TimeConst",
    "ConfScorer",
    "TimeGapDecay",
    "MultiTimeScale",
]

class LearnableTimeScale(nn.Module):
    """
    Positive, trainable time scale τ (in seconds), clamped to [min_seconds, max_seconds].
    forward() -> τ tensor (scalar on the current device/dtype).
    """
    def __init__(self, init_seconds=3600.0, trainable=True,
                 min_seconds=60.0, max_seconds=7*24*3600.0):
        super().__init__()
        init_seconds = float(init_seconds)
        self.log_ts = nn.Parameter(
            torch.tensor(math.log(init_seconds), dtype=torch.float32),
            requires_grad=bool(trainable)
        )
        self.min_seconds = float(min_seconds)
        self.max_seconds = float(max_seconds)

    def forward(self) -> torch.Tensor:
        ts = torch.exp(self.log_ts)
        return torch.clamp(ts, self.min_seconds, self.max_seconds)

    @torch.no_grad()
    def set_seconds(self, seconds: float):
        seconds = float(max(self.min_seconds, min(self.max_seconds, seconds)))
        self.log_ts.copy_(torch.tensor(math.log(seconds), dtype=torch.float32))

    def seconds(self) -> float:
        return float(torch.exp(self.log_ts).item())

class TimeConst(nn.Module):
    """
    Time constant τ = softplus(log_ts) + τ_min, then clamped to [τ_min, τ_max].
    This is convenient if you want a strictly-positive parameter with a hard floor.
    """
    def __init__(self, init_tau: float = 3.0, tau_min: float = 1e-3, tau_max: float = 1e3):
        super().__init__()
        init_tau = float(init_tau)
        self.log_ts = nn.Parameter(torch.tensor(math.log(init_tau), dtype=torch.float32))
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

    def forward(self) -> torch.Tensor:
        tau = F.softplus(self.log_ts) + self.tau_min
        return tau.clamp(self.tau_min, self.tau_max)

class ConfScorer(nn.Module):
    """
    Confidence scorer (backward compatible)

    • Same forward contract as your original:
        forward(x, mask) -> conf in [min_conf, 1], shape [B,T,1] or [B,1]
        - preserves coverage-aware floor
        - optional EMA smoothing over time

    • NEW (optional) self-supervised calibration:
        ssl_loss(x, mask, xhat, conf_pred=None, q=0.10, scale=1.0, reduction="mean")
      - randomly hides a small fraction (q) of truly observed entries
      - computes reconstruction error from xhat on those entries
      - builds a soft confidence target: conf_tgt = exp(-mean_abs_error)
      - trains conf to match conf_tgt only at the sampled steps
    """
    def __init__(self,
                 dim: int,
                 hidden: int = 64,
                 min_conf: float = 1e-3,
                 conf_from: str = "x+mask",     # {"x","mask","x+mask"}
                 smooth_ema: float | None = None,
                 dropout: float = 0.0,
                 use_layernorm: bool = False):
        super().__init__()
        assert conf_from in ("x", "mask", "x+mask")
        self.dim = int(dim)
        self.min_conf = float(min_conf)
        self.conf_from = conf_from
        self.smooth_ema = float(smooth_ema) if smooth_ema is not None else None

        net_in = self.dim if conf_from in ("x", "mask") else 2 * self.dim
        layers = [nn.Linear(net_in, hidden), nn.SiLU()]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(hidden, 1))  # per-step logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [B,T,D] or [B,D]
        mask : [B,T,D] or [B,D] or [B,T] / [B,T,1]
        returns conf : [B,T,1] or [B,1]
        """
        squeeze_back = False

        # Coerce to [B,T,D] for scoring
        if x.dim() == 2:   # [B,D] -> [B,1,D]
            x = x.unsqueeze(1)
            squeeze_back = True
        B, T, D = x.shape

        # Broadcast mask to [B,T,D]
        if mask.dim() == 2:           # [B,D] -> [B,1,D] -> [B,T,D] (T=1)
            mask = mask.unsqueeze(1).expand(B, T, D)
        elif mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.expand(B, T, D)
        elif mask.dim() == 3 and mask.size(-1) == D:
            pass
        else:
            raise ValueError(f"mask must be [B,T,D], [B,T,1], or [B,D]; got {tuple(mask.shape)}")

        # Sanitize
        x_s = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m   = torch.nan_to_num(mask, nan=0.0).clamp(0.0, 1.0)
        miss= (1.0 - m)

        # Features for the scorer (does NOT modify x elsewhere)
        if self.conf_from == "x":
            feat = x_s
        elif self.conf_from == "mask":
            feat = miss
        else:
            feat = torch.cat([x_s, miss], dim=-1)  # "x+mask"

        # Per-step logits -> sigmoid -> scale to [min_conf, 1]
        conf = torch.sigmoid(self.net(feat))                   # [B,T,1]
        conf = conf * (1.0 - self.min_conf) + self.min_conf

        # Coverage-aware floor: if any feature observed at that step,
        # ensure conf >= mean coverage at that step.
        coverage = m.mean(dim=-1, keepdim=True)                # [B,T,1]
        step_obs = (m.sum(dim=-1, keepdim=True) > 0)           # [B,T,1] boolean
        conf = torch.where(step_obs, torch.maximum(conf, coverage), conf)

        # Optional EMA smoothing over time
        if self.smooth_ema is not None and T > 1:
            lam = self.smooth_ema
            c_out = [conf[:, 0:1, :]]
            c_prev = c_out[0]
            for t in range(1, T):
                c_prev = lam * c_prev + (1.0 - lam) * conf[:, t:t+1, :]
                c_out.append(c_prev)
            conf = torch.cat(c_out, dim=1)

        if squeeze_back:
            return conf.squeeze(1)  # [B,1]
        return conf                 # [B,T,1]

    @staticmethod
    def _ensure_seq(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Make x shape [B,T,D]; return (x_seq, squeezed)."""
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeezed = True
        return x, squeezed

    @torch.no_grad()
    def _build_ssl_targets(self,
                           x: torch.Tensor,      # [B,T,D]
                           mask: torch.Tensor,   # [B,T,D]
                           xhat: torch.Tensor,   # [B,T,D]
                           q: float,
                           scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          conf_tgt : [B,T,1]  soft target in (0,1]
          step_msk : [B,T,1]  where to apply the loss (at least one sampled obs in step)
        """
        B, T, D = x.shape
        obs = (mask > 0.5)

        # sample a small extra mask only on observed entries
        extra = (torch.rand(B, T, D, device=x.device) < q)
        tgt_mask = obs & extra

        # step-wise mean absolute error on sampled entries
        err = (x - xhat).abs()
        err = err / (float(scale) + 1e-6)

        cnt_raw  = tgt_mask.sum(dim=-1, keepdim=True)            # [B,T,1]
        cnt_safe = cnt_raw.clamp_min(1.0)                        # for division only
        step_err = (err * tgt_mask).sum(dim=-1, keepdim=True) / cnt_safe
        conf_tgt = torch.exp(-step_err).clamp(1e-6, 1.0)
        step_msk = (cnt_raw > 0).to(conf_tgt.dtype)              # real “where-to-apply”
        return conf_tgt, step_msk

    def ssl_loss(self,
                 x: torch.Tensor,          # [B,T,D] or [B,D]
                 mask: torch.Tensor,       # [B,T,D] or [B,D] / [B,T] / [B,T,1]
                 xhat: torch.Tensor,       # [B,T,D] or [B,D]  (from DecayAware)
                 conf_pred: torch.Tensor | None = None,  # [B,T,1] or [B,1]
                 *,
                 q: float = 0.10,
                 scale: float = 1.0,
                 reduction: str = "mean") -> torch.Tensor:
        """
        Self-supervised calibration loss for confidence.
        Typical usage after DecayAware:
            xhat_seq, conf_seq = da.forward(x, mask, delta)
            loss += 0.05 * scorer.ssl_loss(x, mask, xhat_seq, conf_pred=conf_seq, q=0.10)
        """
        x, _ = self._ensure_seq(x)
        xhat, _ = self._ensure_seq(xhat)

        # broadcast mask to [B,T,D]
        if mask.dim() == 2:       # [B,D] -> [B,1,D] -> [B,T,D]
            mask = mask.unsqueeze(1).expand_as(x)
        elif mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.expand_as(x)
        elif mask.dim() == 3 and mask.size(-1) == x.size(-1):
            pass
        else:
            raise ValueError(f"mask must be [B,T,D], [B,T,1], or [B,D]; got {tuple(mask.shape)}")

        # targets and where-to-apply mask
        conf_tgt, step_msk = self._build_ssl_targets(x, mask, xhat, q=float(q), scale=float(scale))  # [B,T,1] each

        # predicted conf (reuse if provided)
        if conf_pred is None:
            conf_pred = self.forward(x, mask)  # [B,T,1]
        else:
            if conf_pred.dim() == 2:   # [B,1] -> [B,1,1] -> [B,T,1]
                conf_pred = conf_pred.unsqueeze(1).expand(conf_tgt.size(0), conf_tgt.size(1), 1)
            elif conf_pred.dim() == 3 and conf_pred.size(1) == 1 and conf_tgt.size(1) > 1:
                conf_pred = conf_pred.expand(conf_tgt.size(0), conf_tgt.size(1), 1)

        # MSE only where we sampled targets
        se = (conf_pred - conf_tgt).pow(2) * step_msk
        if reduction == "none":
            return se
        denom = step_msk.sum().clamp_min(1.0)
        return se.sum() / denom

class TimeGapDecay(nn.Module):
    """
    gamma = exp(-softplus(W * d + b)) ∈ (0,1], strictly decreasing w.r.t. nonnegative delta d.
    - Accepts d shaped [..., in_dim] (often in_dim=1 for a scalar Δ).
    - Maps to [..., out_dim], e.g. D features or H hidden units.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        # Reasonable starting point: mild decay (~0.95)
        nn.init.constant_(self.lin.bias, -3.0)          # softplus(-3) ~ 0.048 → exp(-0.048) ≈ 0.953
        nn.init.uniform_(self.lin.weight, 0.0, 0.05)    # small → softplus(weight) ≈ small positive

    def forward(self, delta_norm: torch.Tensor) -> torch.Tensor:
        # Sanitize: no NaN/Inf, non-negative, clamp large to keep exp() safe
        d = torch.nan_to_num(delta_norm, nan=0.0, posinf=1e3, neginf=0.0)
        d = d.clamp_min(0.0).clamp_max(1e3)

        # Non-negative effective weights via softplus; bias free to move
        W_pos = F.softplus(self.lin.weight)             # [out_dim, in_dim] ≥ 0
        b = self.lin.bias
        y = F.linear(d, W_pos, b)                       # [..., out_dim]

        # γ = exp( -softplus(y) ), then clamp for stability
        rate = F.softplus(y).clamp_min(0.0).clamp_max(30.0)   # prevent overflow in exp
        gamma = torch.exp(-rate).clamp_min(1e-6)              # (0,1]; floor to keep grads alive
        return gamma

class MultiTimeScale(nn.Module):
    def __init__(self, init_seconds=3600.0, l2_pull=1e-3):
        super().__init__()
        self.shared = LearnableTimeScale(init_seconds, trainable=True)
        self.soil   = LearnableTimeScale(init_seconds, trainable=True)
        self.indoor = LearnableTimeScale(init_seconds, trainable=True)
        self.weath  = LearnableTimeScale(init_seconds, trainable=True)
        self.tab    = LearnableTimeScale(init_seconds, trainable=True)
        self.l2_pull = float(l2_pull)
    def penalty(self):
        s = self.shared()
        return self.l2_pull * sum((m() - s).pow(2).mean()
                                  for m in [self.soil, self.indoor, self.weath, self.tab])
