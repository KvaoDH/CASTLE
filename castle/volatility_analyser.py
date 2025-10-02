# castle/volatility_analyser.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helper import TimeConst

__all__ = ["VolatilityAnalyzer", "ResidualReliability"]

class VolatilityAnalyzer(nn.Module):
    """
    Uses only y-anchored inputs to judge reliability (no external confidences).
    Input anchor (required): [ y_t, y_p1, y_p7, d1, d7 ]  → [B,5]
    Optional summary: [B,20] — used only for a leash term (baseline, y_p1, shock).

    Returns:
      v_emb : [B, vol_dim]
      r     : [B,1] in [rel_min_conf, 1]
      diag  : dict with raw features + penalty components + smoothing state
    """
    def __init__(
        self,
        vol_dim: int,
        *,
        z_clip: float = 6.0,
        eps: float = 1e-8,
        hidden: int = 64,
        dropout: float = 0.05,
        rel_hidden: int = 64,
        rel_min_conf: float = 1e-3,
        leash_blend: float = 0.25,
        fp_weight: float = 0.5,
        fn_weight: float = 0.5,
        smooth_weight: float = 0.1,
    ):
        super().__init__()
        self.z_clip = float(z_clip)
        self.eps = float(eps)
        self.rel_min_conf = float(rel_min_conf)
        self.leash_blend = float(max(0.0, min(1.0, leash_blend)))
        self.fp_w = float(fp_weight)
        self.fn_w = float(fn_weight)
        self.sm_w = float(smooth_weight)

        # 9 raw volatility features (includes y_t)
        self.proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vol_dim)
        )
        self.rel_head = nn.Sequential(
            nn.LayerNorm(vol_dim),
            nn.Linear(vol_dim, rel_hidden), nn.SiLU(),
            nn.Linear(rel_hidden, 1), nn.Sigmoid()
        )

        # learned low-pass smoothing for r
        self.r_tau = TimeConst(init_tau=3.0, tau_min=0.5, tau_max=30.0)
        self.register_buffer("_r_state", torch.tensor(0.5))

    def _vol_raw(self, anchor: torch.Tensor):
        # anchor: [y_t, y_p1, y_p7, d1, d7]
        assert anchor.dim() == 2 and anchor.size(1) == 5, f"expected [B,5]=[y_t,p1,p7,d1,d7], got {tuple(anchor.shape)}"
        y_t, y_p1, y_p7, d1, d7 = anchor[:, 0:1], anchor[:, 1:2], anchor[:, 2:3], anchor[:, 3:4], anchor[:, 4:5]

        eps = self.eps
        denom1 = y_p1.abs() + eps
        denom7 = y_p7.abs() + eps

        r1 = (torch.nan_to_num(d1, 0.0).abs() / denom1).clamp(0.0, self.z_clip)
        r7 = (torch.nan_to_num(d7, 0.0).abs() / denom7).clamp(0.0, self.z_clip)

        vol_level  = 0.5 * (r1 + r7)                                            # magnitude
        vol_change = (r1 - r7)                                                  # acceleration
        vol_z      = ((r1 - r7) / (r7.abs() + eps)).clamp(-self.z_clip, self.z_clip)

        # directional agreement of d1 and d7
        dir_agree  = torch.tanh((torch.nan_to_num(d1,0.0) * torch.nan_to_num(d7,0.0)) /
                                (torch.nan_to_num(d1,0.0).abs() + torch.nan_to_num(d7,0.0).abs() + eps))

        # price/level discrepancy proxies
        ratio_dev_p = torch.tanh((y_p1 - y_p7).abs() / (y_p7.abs() + eps))      # ~|1 - y_p1/y_p7|
        r_t7        = ((y_t - y_p7).abs() / (y_p7.abs() + eps)).clamp(0.0, self.z_clip)
        reverb      = torch.tanh(((y_t - y_p1) * torch.nan_to_num(d7,0.0)) /
                                 ((y_t - y_p1).abs() + torch.nan_to_num(d7,0.0).abs() + eps))

        # 9 features
        return torch.cat([vol_level, vol_change, vol_z, r1, r7, dir_agree, ratio_dev_p, r_t7, reverb], dim=1)

    def _leash_from_summary(self, raw_vol: torch.Tensor, summary: torch.Tensor | None):
        if summary is None: return None
        vol_level  = raw_vol[:, 0:1]
        baseline   = summary[:, 0:1]   # feat[0]
        y_today    = summary[:, 7:8]   # feat[7]
        shock      = summary[:, 14:15] # diagnostic shock
        gap   = (y_today - baseline).abs()
        norm  = (vol_level + 1e-8)
        drift = (gap / norm) + shock.abs()
        # drift↑ ⇒ leash↑ ⇒ r↑
        return torch.sigmoid(torch.clamp(drift - 0.5, -8.0, 8.0))

    def _batch_thresholds(self, vol_level: torch.Tensor):
        # robust per-batch thresholds for "high" / "low" volatility
        v = vol_level.detach().view(-1)
        med = v.median()
        mad = (v - med).abs().median() + 1e-6
        high_t = med + 2.0 * mad
        low_t  = med + 0.5 * mad
        return high_t, low_t

    def forward(self, anchor: torch.Tensor, summary: torch.Tensor | None = None):
        raw_vol = self._vol_raw(anchor)
        v_emb   = self.proj(raw_vol)

        # base reliability from embedding
        r_conf  = self.rel_head(v_emb) * (1.0 - self.rel_min_conf) + self.rel_min_conf

        # learned smoothing toward running scalar state
        tau = self.r_tau()
        alpha = torch.exp(-1.0 / tau).clamp(0.0, 0.999)
        with torch.no_grad():
            batch_mean = r_conf.detach().mean()
            self._r_state.mul_(alpha).add_((1.0 - alpha) * batch_mean)
        r_smooth = (1.0 - alpha) * r_conf + alpha * self._r_state

        # optional leash (baseline/y_p1/shock only; NOT confidences)
        leash   = self._leash_from_summary(raw_vol, summary)
        r = (1.0 - self.leash_blend) * r_smooth + (self.leash_blend * leash if leash is not None else 0.0)
        r = r.clamp(self.rel_min_conf, 1.0)

        # --- penalties (unsupervised, batch-robust) ---
        vol_level = raw_vol[:, 0:1]
        high_t, low_t = self._batch_thresholds(vol_level)
        high_mask = torch.sigmoid(5.0 * (vol_level - high_t))
        low_mask  = torch.sigmoid(5.0 * (low_t - vol_level))

        penalty_fp = (r * low_mask).mean()           # high r while low vol
        penalty_fn = ((1.0 - r) * high_mask).mean()  # low r while high vol
        penalty_smooth = (r - self._r_state).pow(2).mean()

        reg_loss = self.fp_w * penalty_fp + self.fn_w * penalty_fn + self.sm_w * penalty_smooth

        diag = {
            "raw_vol": raw_vol.detach(),
            "r_conf": r_conf.detach(),
            "r_smooth": r_smooth.detach(),
            "r_state": self._r_state.detach().clone(),
            "leash": (None if leash is None else leash.detach()),
            "penalty_fp": penalty_fp.detach(),
            "penalty_fn": penalty_fn.detach(),
            "penalty_smooth": penalty_smooth.detach(),
            "reg_loss": reg_loss
        }
        return v_emb, r, diag

class ResidualReliability(nn.Module):
    """
    Turn past absolute residuals into a reliability r_resid ∈ (0,1].
    - past_abs_residuals: [B, k] of |y_true - y_pred| from strictly previous days
    - scale_hint        : [B, 1] optional magnitude (e.g., baseline |y|), for scale invariance
    """
    def __init__(self, k: int = 7, gain: float = 0.35, clip: float = 6.0, eps: float = 1e-6):
        super().__init__()
        self.k = int(k)
        self.gain = float(gain)     # blend weight toward residual reliability
        self.clip = float(clip)     # cap normalized residuals
        self.eps = float(eps)

    def forward(self, past_abs_residuals: torch.Tensor, scale_hint: torch.Tensor | None = None) -> torch.Tensor:
        # normalize by scale if provided
        if scale_hint is not None:
            s = torch.nan_to_num(scale_hint, nan=1.0).abs().clamp_min(self.eps)  # [B,1]
            res = past_abs_residuals / s
        else:
            res = past_abs_residuals
        res = torch.nan_to_num(res, nan=0.0, posinf=self.clip, neginf=0.0).clamp_max(self.clip)
        r_resid = 1.0 / (1.0 + res.mean(dim=1, keepdim=True))   # [B,1] ∈ (0,1]
        return r_resid.clamp(0.01, 1.0)

    def blend(self, r_vol: torch.Tensor, r_resid: torch.Tensor) -> torch.Tensor:
        # monotone pull toward residual reliability; gain in [0,1]
        return r_vol + self.gain * (r_resid - r_vol)
