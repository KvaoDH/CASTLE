# castle/regime.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RegimeAdapter"]

class RegimeAdapter(nn.Module):
    """
    Inputs:
      y_core : [B,1]        (typically the anchor baseline)
      summary: [B,20]       (conditioning/routing; baseline is summary[:,0])
      r      : [B,1]        reliability from VolatilityAnalyzer
      regime_memory: dict   (optional). If provided and update_memory=True,
                             will be updated in-place with EMA's of a and lambda.

    Behavior:
      - lambda is capped by lambda_max * r (reliability cap).
      - small memory-based inertia on a & lambda to stabilize across steps.
      - can use internal EMA memory if `enable_memory()` was called.
    """
    def __init__(
        self,
        cond_dim: int = 20,
        K: int = 6,
        a_span: float = 0.25,
        lambda_max: float = 0.80,
        lambda_init: float = 0.20,
        gate_temp: float = 1.0,
        gate_entropy_reg: float = 0.0,
        use_cond_norm: bool = True,
        mem_beta: float = 0.9,  # EMA factor for memory smoothing
    ):
        super().__init__()
        self.K = int(K)
        self.a_span = float(max(0.0, a_span))
        self.lambda_max = float(max(0.0, lambda_max))
        self.gate_temp = float(max(1e-3, gate_temp))
        self.gate_entropy_reg = float(max(0.0, gate_entropy_reg))
        self.use_cond_norm = bool(use_cond_norm)
        self.cond_norm = nn.LayerNorm(cond_dim) if self.use_cond_norm else nn.Identity()
        self.mem_beta = float(max(0.0, min(0.999, mem_beta)))

        self.gate = nn.Sequential(nn.Linear(cond_dim, 32), nn.ReLU(), nn.Linear(32, self.K))
        self.to_a = nn.Linear(cond_dim, self.K)
        self.to_lambda = nn.Linear(cond_dim, self.K)
        nn.init.zeros_(self.to_a.weight); nn.init.zeros_(self.to_a.bias)
        nn.init.zeros_(self.to_lambda.weight)
        with torch.no_grad():
            self.to_lambda.bias.fill_(math.log(max(1e-6, lambda_init) / max(1e-6, 1.0 - lambda_init)))

        self.ctrl = nn.Sequential(nn.Linear(cond_dim, 32), nn.SiLU(), nn.Linear(32, 3))

        # indices for summary parsing
        self.idx = {"feat": slice(0,10), "mix": slice(10,14), "shock": 14,
                    "conf_vec": slice(15,19), "conf_glob": 19, "baseline": 0}

        # optional internal memory (enabled via enable_memory())
        self._use_internal_memory = False
        self._memory: dict[str, torch.Tensor] | None = None

    # -------- memory helpers --------
    def enable_memory(self):
        self._use_internal_memory = True
        if self._memory is None:
            self._memory = {}

    def disable_memory(self):
        self._use_internal_memory = False

    @staticmethod
    def _mem_get(mem: dict | None, key: str, like: torch.Tensor):
        if (mem is None) or (key not in mem) or (mem[key] is None):
            return torch.zeros_like(like)
        t = mem[key]
        return t.to(dtype=like.dtype, device=like.device).expand_as(like)

    @staticmethod
    def _mem_set(mem: dict | None, key: str, value: torch.Tensor, detach: bool = True):
        if mem is None: return
        mem[key] = (value.detach() if detach else value)

    # -------- parsing --------
    def _parse_summary(self, summary: torch.Tensor):
        i = self.idx
        feat      = summary[:, i["feat"]]
        baseline  = feat[:, i["baseline"]:i["baseline"]+1]
        return feat, baseline

    # -------- forward --------
    def forward(
        self,
        *,
        y_core: torch.Tensor,
        summary: torch.Tensor,
        r: torch.Tensor | None,
        regime_memory: dict | None = None,
        update_memory: bool = True,
        return_controls: bool = False
    ):
        if r is None:
            return y_core if not return_controls else {
                "y_cal": y_core, "gate_w": None, "a": None, "lambda": None,
                "reliability": None, "memory": regime_memory,
                "router_entropy": torch.tensor(0.0, device=y_core.device, dtype=y_core.dtype),
                "aux_reg": torch.tensor(0.0, device=y_core.device, dtype=y_core.dtype),
                "last_reg": torch.tensor(0.0, device=y_core.device, dtype=y_core.dtype),
            }

        dtype = y_core.dtype
        device = y_core.device

        feat, baseline = self._parse_summary(summary)
        cond = self.cond_norm(summary.to(device=device, dtype=dtype))

        # router
        logits = self.gate(cond).to(dtype) / self.gate_temp
        w = F.softmax(logits, dim=1).to(dtype)  # [B,K]
        router_ent = (-(w * (w.clamp_min(1e-8).log())).sum(dim=1)).mean()

        # experts
        a_raw   = self.to_a(cond).to(dtype)
        lam_raw = self.to_lambda(cond).to(dtype)
        a   = 1.0 + self.a_span * torch.tanh(a_raw)

        r = torch.nan_to_num(r, nan=0.0).clamp(0.0, 1.0)      # [B,1]
        lam_cap = (self.lambda_max * r).to(dtype)
        lam = lam_cap * torch.sigmoid(lam_raw)                 # [B,K]

        # --- choose memory dict ---
        mem = regime_memory
        if mem is None and self._use_internal_memory:
            if self._memory is None:
                self._memory = {}
            mem = self._memory

        # --- memory smoothing (EMA inertia on means of a & lambda) ---
        a_mean  = a.detach().mean(dim=1, keepdim=True)   # [B,1]
        l_mean  = lam.detach().mean(dim=1, keepdim=True) # [B,1]
        a_ema_prev = self._mem_get(mem, "a_ema", a_mean)
        l_ema_prev = self._mem_get(mem, "lam_ema", l_mean)
        a_ema = self.mem_beta * a_ema_prev + (1.0 - self.mem_beta) * a_mean
        l_ema = self.mem_beta * l_ema_prev + (1.0 - self.mem_beta) * l_mean
        if update_memory:
            self._mem_set(mem, "a_ema", a_ema)
            self._mem_set(mem, "lam_ema", l_ema)

        # apply gentle inertia toward the EMAs
        a = 0.85 * a + 0.15 * a_ema.expand_as(a)
        lam = 0.75 * lam + 0.25 * l_ema.expand_as(lam)

        # combine
        yc = y_core.to(dtype).expand(-1, self.K)
        an = baseline.to(dtype).expand(-1, self.K)
        yk = a * yc + lam * (an - yc)
        y_cal = (w * yk).sum(dim=1, keepdim=True)

        last_reg = (a - 1.0).pow(2).mean() + 0.1 * (lam.pow(2).mean())
        aux_reg = self.gate_entropy_reg * (-router_ent)

        if not return_controls:
            return y_cal

        ctrl = torch.sigmoid(self.ctrl(cond))
        out = {
            "y_cal": y_cal, "gate_w": w, "a": a, "lambda": lam,
            "reliability": r.to(dtype),
            "memory": mem,
            "leash_scale": ctrl[:, 0:1], "adapt_gain": ctrl[:, 1:2], "stability": ctrl[:, 2:3],
            "router_entropy": router_ent.detach(),
            "aux_reg": torch.as_tensor(aux_reg, device=y_cal.device, dtype=y_cal.dtype),
            "last_reg": torch.as_tensor(last_reg, device=y_cal.device, dtype=y_cal.dtype),
        }
        return out
