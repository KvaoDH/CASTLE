# castle/head.py
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["ConsensusTrustHead"]

class ConsensusTrustHead(nn.Module):
    """
    Symmetric, responsibility-balanced combiner for (y_core from fused) and (y_reg from regime).

    Inputs (forward):
      fused: [B, F]                 # features for the core predictor and its variance head
      regime_y: [B,1] or None       # calibrated anchor/regime lane prediction
      regime_r: [B,1] or None       # reliability in [0,1]
      shock: [B,1] or None          # shock magnitude proxy in [-1,1] (we use abs)
      conf_fuse: [B,1] or None      # fusion confidence in [0,1]
      scale_hint: [B,1] or None     # magnitude scale for disagreement normalization
      return_alpha: bool            # kept for API parity (alpha=anchor share)
    """
    def __init__(
        self,
        fused_dim: int,
        base_hidden: int = 128,
        *,
        delta_clip: float = 6.0,
        dropout: float = 0.10,
        use_hetero: bool = True,      # enable core variance head
        tau_core: float = 0.90,       # cap on core lane dominance (0..1]
        tau_reg: float = 0.90,        # cap on regime lane dominance (0..1]
        # regime calibration constraints
        reg_gain_max: float = 0.15,   # y_reg = (1+γ)*regime_y + β, with |γ|<=reg_gain_max
        reg_bias_max: float = 0.50,   # |β|<=reg_bias_max
        # disagreement inflation strengths
        core_inflate: float = 0.60,
        reg_inflate: float  = 0.60,
        # confidence/shock sensitivities
        core_conf_gain: float = 0.50, # inflate core var when conf_fuse low
        reg_shock_gain: float = 0.50  # inflate regime var when |shock| high
    ):
        super().__init__()
        self.fused_dim = int(fused_dim)
        self.delta_clip = float(max(1.0, delta_clip))
        self.use_hetero = bool(use_hetero)

        # ---------- Core predictor ----------
        self.base = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, base_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(base_hidden, base_hidden), nn.SiLU(),
            nn.Linear(base_hidden, 1),
        )

        # Core variance head (log-variance)
        self.var_head = None
        if self.use_hetero:
            self.var_head = nn.Sequential(
                nn.LayerNorm(self.fused_dim),
                nn.Linear(self.fused_dim, base_hidden), nn.SiLU(),
                nn.Linear(base_hidden, 1)
            )

        # Learnable inflation gains are fixed by config constants above; keep scalars in buffers
        self.register_buffer("_core_inflate", torch.tensor(float(core_inflate)))
        self.register_buffer("_reg_inflate",  torch.tensor(float(reg_inflate)))
        self.register_buffer("_core_conf_gain", torch.tensor(float(core_conf_gain)))
        self.register_buffer("_reg_shock_gain", torch.tensor(float(reg_shock_gain)))

        # Temperature caps (constants)
        self.register_buffer("_tau_core", torch.tensor(float(max(0.0, min(1.0, tau_core)))))
        self.register_buffer("_tau_reg",  torch.tensor(float(max(0.0, min(1.0, tau_reg)))))

        # ---------- Regime light calibration ----------
        # y_reg = (1 + γ)*regime_y + β, with |γ|<=reg_gain_max, |β|<=reg_bias_max
        self.reg_gain_raw = nn.Parameter(torch.zeros(1))
        self.reg_bias_raw = nn.Parameter(torch.zeros(1))
        self.register_buffer("_reg_gain_max", torch.tensor(float(abs(reg_gain_max))))
        self.register_buffer("_reg_bias_max", torch.tensor(float(abs(reg_bias_max))))

        # Diagnostics cache
        self._last = {}

    def _norm_disagreement(self, y_core, y_reg, scale_hint):
        delta = (y_reg - y_core).abs()            # [B,1]
        if scale_hint is None:
            s = y_core.abs().clamp_min(1e-6)
        else:
            s = scale_hint.abs().clamp_min(1e-6)
        delta_norm = (delta / s).clamp_max(self.delta_clip)
        return delta, delta_norm

    def _calibrate_regime(self, regime_y):
        # constrain gain ∈ [1-reg_gain_max, 1+reg_gain_max]; bias ∈ [-reg_bias_max, +reg_bias_max]
        g = torch.tanh(self.reg_gain_raw) * self._reg_gain_max   # in [-max, max]
        b = torch.tanh(self.reg_bias_raw) * self._reg_bias_max
        y_reg = (1.0 + g) * regime_y + b
        return y_reg, g.detach(), b.detach()

    def forward(
        self,
        *,
        fused: torch.Tensor,
        regime_y: torch.Tensor | None = None,
        regime_r: torch.Tensor | None = None,
        shock: torch.Tensor | None = None,
        conf_fuse: torch.Tensor | None = None,
        scale_hint: torch.Tensor | None = None,
        return_alpha: bool = True  # kept for parity; alpha = anchor share
    ):
        # ----------- core forecast -----------
        y_core = self.base(fused)                        # [B,1]

        # Core base log-variance (or a constant if hetero disabled)
        if self.use_hetero and (self.var_head is not None):
            logv_core = self.var_head(fused).clamp(-10.0, 5.0)
        else:
            logv_core = torch.zeros_like(y_core)

        # If no regime lane, just return core
        if (regime_y is None) or (regime_r is None):
            out = y_core
            aux = {
                "alpha": torch.zeros_like(out),   # anchor share
                "w_core": torch.ones_like(out), "w_reg": torch.zeros_like(out),
                "y_core": y_core.detach(), "y_reg": torch.zeros_like(out),
                "y_final": out.detach(),
                "logv_core": logv_core.detach(), "logv_reg": torch.zeros_like(out),
                "delta": torch.zeros_like(out), "delta_norm": torch.zeros_like(out),
            }
            self._last = aux
            return out, aux

        # ----------- regime light calibration -----------
        y_reg, reg_gain, reg_bias = self._calibrate_regime(regime_y)

        # ----------- disagreement -----------
        delta, delta_norm = self._norm_disagreement(y_core, y_reg, scale_hint)

        # ----------- regime uncertainty -----------
        # Start from reliability (higher r => lower variance), modulate with shock and disagreement.
        r = regime_r.clamp(1e-4, 1.0 - 1e-4)
        base_reg_var = torch.log1p((1.0 - r))            # small when r≈1, larger when r low
        sh = (shock.abs() if shock is not None else torch.zeros_like(y_reg)).clamp(0.0, 1.0)
        # Effective log-variance for regime
        logv_reg = (base_reg_var
                    + self._reg_shock_gain * sh
                    + self._reg_inflate * delta_norm).clamp(-10.0, 6.0)

        # ----------- core uncertainty inflation -----------
        cf = (conf_fuse if conf_fuse is not None else torch.zeros_like(y_core)).clamp(0.0, 1.0)
        logv_core_eff = (logv_core
                         + self._core_conf_gain * (1.0 - cf)
                         + self._core_inflate * delta_norm).clamp(-10.0, 6.0)

        # ----------- precisions with temperature caps -----------
        pi_core = self._tau_core * torch.exp(-logv_core_eff)   # [B,1]
        pi_reg  = self._tau_reg  * torch.exp(-logv_reg)        # [B,1]
        denom = (pi_core + pi_reg).clamp_min(1e-9)
        w_core = pi_core / denom
        w_reg  = pi_reg  / denom

        # ----------- consensus -----------
        y = w_core * y_core + w_reg * y_reg

        aux = {
            # keep alpha semantics = anchor contribution
            "alpha": w_reg.detach(),
            "w_core": w_core.detach(), "w_reg": w_reg.detach(),
            "y_core": y_core.detach(), "y_reg": y_reg.detach(),
            "y_final": y.detach(),
            "logv_core": logv_core_eff.detach(), "logv_reg": logv_reg.detach(),
            "delta": delta.detach(), "delta_norm": delta_norm.detach(),
            "reg_calib": {"gain": reg_gain, "bias": reg_bias},
        }
        self._last = aux
        return y, aux
