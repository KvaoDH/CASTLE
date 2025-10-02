# castle/anchor.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import TimeConst

__all__ = ["AnchorCreator"]

class AnchorCreator(nn.Module):
    """
    Input anchor (required): [ y_today, y_prev1, y_prev7, y_diff1, y_diff7 ]  → [B,5]

    Creates a leak-safe y-only baseline using 4 candidates (shape preserved):
      candidates = [y_fast(y_t, y_p1), ema3, ema7, y_p7]
      where y_fast blends y_t into y_p1 with a shock-aware, time-constant-controlled weight.

    Returns (contracts kept stable for downstream):
      baseline : [B,1]
      feat     : [B,10] = [baseline, y_p1, y_p7, ema3, ema7, d1, d7, y_t, 0, ratio_1_7]
      mix      : [B,4]  over [y_fast, ema3, ema7, y_p7]
      conf_vec : [B,4]  = ones (neutral)
      conf_glob: [B,1]  = ones
      shock    : [B,1]  in [-1,1] (diagnostic only)
      summary  : [B,20] = feat(10) ⊕ mix(4) ⊕ shock(1) ⊕ conf_vec(4) ⊕ conf_glob(1)
      debug    : dict
    """
    def __init__(
        self,
        *,
        shock_gain: float = 1.0,
        temperature: float = 1.0,
        tau_init: float = 3.0,
        tau_min: float = 1e-3,
        tau_max: float = 1e3,
    ):
        super().__init__()
        self.shock_gain = float(shock_gain)

        # Prior from [1, shock] → 4 logits (small nudge only)
        self.logit_affine = nn.Linear(2, 4, bias=True)

        # Simple shock detector using only y-anchored deltas (includes y_t surprise)
        self.shock_mlp = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(inplace=True), nn.Linear(8, 1)
        )

        # Softmax temperature (>=0.5 for stability)
        self._temp = nn.Parameter(torch.tensor([max(1e-3, float(temperature))]))

        # Time constant for fast-candidate mixing
        self.tau_param = TimeConst(init_tau=float(tau_init),
                                   tau_min=float(tau_min),
                                   tau_max=float(tau_max))

    @staticmethod
    def _split(anchor: torch.Tensor):
        assert anchor.dim() == 2 and anchor.size(1) == 5, f"expected [B,5]=[y_t,p1,p7,d1,d7], got {tuple(anchor.shape)}"
        y_t  = anchor[:, 0:1]
        y_p1 = anchor[:, 1:2]
        y_p7 = anchor[:, 2:3]
        d1   = anchor[:, 3:4]
        d7   = anchor[:, 4:5]
        return y_t, y_p1, y_p7, d1, d7

    @staticmethod
    def _safe_ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
        a0 = torch.nan_to_num(a, nan=0.0)
        b0 = torch.nan_to_num(b, nan=0.0)
        ok = torch.isfinite(a) & torch.isfinite(b) & (b0.abs() > eps)
        return torch.where(ok, a0 / (b0 + eps), torch.ones_like(a0))

    def forward(self, anchor: torch.Tensor):
        B = anchor.size(0)
        device, dtype = anchor.device, anchor.dtype

        y_t, y_p1, y_p7, d1, d7 = self._split(anchor)

        # Causal EMA proxies (prev/diff only; no peek ahead)
        ema3 = y_p1 + (1.0/3.0) * torch.nan_to_num(d1, nan=0.0)
        ema7 = y_p7 + (1.0/7.0) * torch.nan_to_num(d7, nan=0.0)

        # Availability masks
        m_p1 = torch.isfinite(y_p1).float()
        m_p7 = torch.isfinite(y_p7).float()
        m_e3 = m_p1.clone()
        m_e7 = m_p7.clone()

        ratio = self._safe_ratio(y_p1, y_p7)
        ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

        # Shock proxy: |d1|,|d7|,|1-ratio|, and today's surprise vs p1
        shock_in = torch.cat([
            torch.abs(torch.nan_to_num(d1, nan=0.0)),
            torch.abs(torch.nan_to_num(d7, nan=0.0)),
            torch.abs(1.0 - ratio),
            torch.abs(torch.nan_to_num(y_t - y_p1, nan=0.0)),
        ], dim=1)
        shock = torch.tanh(self.shock_gain * self.shock_mlp(shock_in))  # [-1,1]

        # TimeConst prior: tau → alpha reactivity
        tau   = self.tau_param().to(dtype=dtype, device=device).clamp_min(1e-6)
        alpha = (1.0 / (1.0 + tau)).clamp(0.0, 1.0)  # tau≈0 → fast, tau→∞ → slow

        # Shock-aware weight for y_t inside the fast candidate
        gamma_t = (alpha * (1.0 - shock.abs())).clamp(0.0, 1.0)  # [B,1]

        # Fast candidate: blend of p1 and today
        y_fast = (1.0 - gamma_t) * y_p1 + gamma_t * y_t

        # 4 candidates (shape preserved)
        cand  = torch.cat([y_fast, ema3, ema7, y_p7], dim=1)  # [B,4]

        # Availability: y_fast needs p1 & y_t → use p1's mask (y_t is required anyway)
        m_fast = m_p1
        cmask  = torch.cat([m_fast, m_e3, m_e7, m_p7], dim=1)  # [B,4]

        # Prior over 4 candidates (time-bias + small shock bias)
        time_bias = torch.tensor([+1.25, +0.25, -0.25, -1.0], device=device, dtype=dtype) * (2.0*alpha - 1.0)
        time_bias = time_bias.view(1, 4).expand(B, -1)

        bias_from_shock = self.logit_affine(torch.cat([torch.ones_like(shock), shock], dim=1))  # [B,4]
        logits = bias_from_shock + time_bias
        logits = logits + (cmask - 1.0) * 1e6  # mask-out unavailable

        temp = F.softplus(self._temp) + 0.5
        mix = F.softmax(logits / temp, dim=1)  # [B,4]
        mix = (mix * cmask)
        mix = mix / mix.sum(dim=1, keepdim=True).clamp_min(1e-8)

        baseline = (mix * torch.nan_to_num(cand, nan=0.0)).sum(dim=1, keepdim=True)  # [B,1]

        zeros = torch.zeros(B, 1, device=device, dtype=dtype)
        feat = torch.cat([
            baseline, y_p1, y_p7, ema3, ema7,
            torch.nan_to_num(d1, nan=0.0),
            torch.nan_to_num(d7, nan=0.0),
            y_t, zeros,                    # <— y_t sits at feat[7]
            ratio
        ], dim=1)

        # Neutral confidences (kept only for 20-D contract)
        conf_vec  = torch.ones(B, 4, device=device, dtype=dtype)
        conf_glob = torch.ones(B, 1, device=device, dtype=dtype)

        summary = torch.cat([feat, mix, shock, conf_vec, conf_glob], dim=1)
        feat    = torch.nan_to_num(feat,    nan=0.0, posinf=0.0, neginf=0.0)
        summary = torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)

        debug = {"mix": mix, "logits": logits, "shock": shock, "tau": tau.detach(), "anchor_raw": anchor}
        return baseline, feat, mix, conf_vec, conf_glob, shock, summary, debug
