# castle/fusion.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FusionAllStream"]

class FusionAllStream(nn.Module):
    """
    Fusion of stream representations with per-stream conditions and confidences.

    • Per-stream scoring: each stream i gets its own feature f_i and its own
      condition slice c_i (from StreamCondBuilder) + its confidence scalar.
    • Confidence is used twice: (1) as an input to the scorer, and
      (2) as a soft bias on logits (bounded).
    • Produces a compact `cond_for_head` = [flattened per-stream cond | entropy | max_w | conf_wmean].

    Inputs
      features        : list of [B, D_i] per stream (finite; e.g., soil, indoor, weather, tabular)
      cond_per_stream : [B, N, Cps] per-stream condition (preferred; with StreamCondBuilder Cps=5)
      cond            : [B, C] global condition (used only if cond_per_stream is None)
      stream_conf     : [B, N] or [B,1] or None per-stream confidences in [0,1]
      stream_mask     : [B, N] optional mask (1=keep, 0=drop)

    Returns
      fused           : [B, sum(D_i)] weighted concat of per-stream features
      aux             : {
                           "weights": [B,N],
                           "entropy": scalar tensor (mean over batch),
                           "max_w":   scalar tensor (mean over batch),
                           "logits":  [B,N],
                           "conf_used":[B,N],
                           "cond_for_head":[B, N*Cps + 3]
                        }
    """
    def __init__(self,
                 input_dims,
                 fusion_dim,
                 *,
                 per_stream_cond_dim: int | None = None,  # set to 5 when using StreamCondBuilder
                 init_mix=0.20,
                 init_temp=1.0,
                 per_stream_norm=True,
                 topk: int | None = None,
                 dropout: float = 0.0,
                 gain_max: float = 0.3):
        super().__init__()
        self.input_dims = list(map(int, input_dims))
        self.n_streams = len(self.input_dims)
        self.fusion_dim = int(fusion_dim)
        self.per_stream_cond_dim = (None if per_stream_cond_dim is None else int(per_stream_cond_dim))
        self.topk_cfg = None if topk is None else int(topk)
        self.topk_enabled = False
        self.per_stream_norm = bool(per_stream_norm)
        self.gain_max = float(gain_max)

        # Per-stream normalization and feature projection to a common hidden
        self.ln_streams = nn.ModuleList([nn.LayerNorm(d) for d in self.input_dims]) if self.per_stream_norm else None
        self.feat_proj = nn.ModuleList([nn.Linear(d, self.fusion_dim) for d in self.input_dims])

        # Per-stream condition projection (+1 for confidence scalar)
        if self.per_stream_cond_dim is not None:
            self.cond_ps_proj = nn.Linear(self.per_stream_cond_dim + 1, self.fusion_dim)  # shared across streams
            self.scorer = nn.Linear(self.fusion_dim, 1)  # shared head for logits
        else:
            # Back-compat global path
            self.attn_proj    = nn.Linear(sum(self.input_dims), self.fusion_dim)
            self.cond_proj    = nn.Identity()
            self.attn_weights = nn.Linear(self.fusion_dim, self.n_streams)
            self._cond_in_dim = 0  # will lazy-init if cond provided

        # Mix with a uniform prior + temperature
        self.register_buffer("mix",  torch.tensor(float(init_mix)))
        self.register_buffer("temp", torch.tensor(float(init_temp)))

        # Confidence bias (learnable scale, bounded with tanh → gain_max)
        self.stream_conf_gain_raw = nn.Parameter(torch.tensor(0.0))

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # diagnostics
        self.last_w = None
        self.last_logits = None
        self.last_entropy = torch.tensor(0.0)
        self.last_maxw = torch.tensor(0.0)

    # ---- runtime toggles ----
    def enable_topk(self, k: int | None = None):
        self.topk_enabled = True
        if k is not None:
            self.topk_cfg = int(k)

    def disable_topk(self):
        self.topk_enabled = False

    # ---- helpers ----
    @staticmethod
    def _safe_logit(p: torch.Tensor) -> torch.Tensor:
        p = torch.nan_to_num(p, nan=0.5, posinf=0.5, neginf=0.5).clamp(1e-4, 1 - 1e-4)
        return torch.log(p) - torch.log(1 - p)

    def _score_per_stream(self, feats, cond_ps, stream_conf):
        B = feats[0].size(0)
        N = self.n_streams
        if stream_conf is None:
            sc = torch.ones(B, N, device=feats[0].device, dtype=feats[0].dtype)
        else:
            sc = stream_conf
            if sc.dim() == 2 and sc.size(1) == 1:
                sc = sc.expand(B, N)
            sc = torch.nan_to_num(sc, nan=0.0).clamp(0.0, 1.0)

        # project features
        f_proj = []
        for i, f in enumerate(feats):
            fi = self.ln_streams[i](f) if self.ln_streams is not None else f
            f_proj.append(self.feat_proj[i](fi))  # [B,F]

        # per-stream cond + confidence → projection
        logits_list = []
        for i in range(N):
            c_i = cond_ps[:, i, :]  # [B,Cps]
            ci_in = torch.cat([c_i, sc[:, i:i+1]], dim=1)  # append confidence
            h = torch.tanh(f_proj[i] + self.cond_ps_proj(ci_in))  # [B,F]
            logit_i = self.scorer(h)  # [B,1]
            logits_list.append(logit_i)
        logits = torch.cat(logits_list, dim=1)  # [B,N]
        return logits, sc

    # ---- forward ----
    def forward(self,
                features,
                cond=None,
                *,
                cond_per_stream: torch.Tensor | None = None,   # preferred: [B,N,Cps]
                stream_mask: torch.Tensor | None = None,       # [B,N], 1=keep, 0=mask
                stream_conf: torch.Tensor | None = None,       # [B,N] or [B,1] or None
                return_aux: bool = True):
        assert isinstance(features, (list, tuple)) and len(features) == self.n_streams, \
            f"expected {self.n_streams} feature tensors, got {len(features)}"
        B = features[0].size(0)

        # ----- Per-stream scorer path (preferred) -----
        if self.per_stream_cond_dim is not None and cond_per_stream is not None:
            assert cond_per_stream.size(1) == self.n_streams and cond_per_stream.size(2) == self.per_stream_cond_dim, \
                f"cond_per_stream must be [B,{self.n_streams},{self.per_stream_cond_dim}]"
            logits, sc = self._score_per_stream(features, cond_per_stream, stream_conf)

        # ----- Back-compat global path -----
        else:
            feats = []
            for i, f in enumerate(features):
                fi = self.ln_streams[i](f) if self.ln_streams is not None else f
                feats.append(fi)
            all_feat = torch.cat(feats, dim=1)  # [B, sum(Di)]

            # lazily create cond_proj to match provided cond size (for drop-in compatibility)
            if cond is None:
                cond = torch.zeros(B, 0, device=all_feat.device, dtype=all_feat.dtype)
            if (not hasattr(self, "_cond_in_dim")) or (self._cond_in_dim != cond.size(1)):
                self._cond_in_dim = cond.size(1)
                self.cond_proj = nn.Linear(self._cond_in_dim, self.fusion_dim).to(all_feat.device)

            h = torch.tanh(self.attn_proj(all_feat) + self.cond_proj(cond))  # [B,F]
            logits = self.attn_weights(h)  # [B,N]
            sc = (stream_conf if stream_conf is not None
                  else torch.ones(B, self.n_streams, device=all_feat.device, dtype=all_feat.dtype))
            if sc.dim() == 2 and sc.size(1) == 1:
                sc = sc.expand(B, self.n_streams)
            sc = torch.nan_to_num(sc, nan=0.0).clamp(0.0, 1.0)

        # ----- confidence bias (soft) -----
        conf_logit = self._safe_logit(sc)           # [B,N]
        conf_gain = self.gain_max * torch.tanh(self.stream_conf_gain_raw)
        if float(conf_gain) != 0.0:
            logits = logits + conf_gain * conf_logit

        # Optional stream mask
        if stream_mask is not None:
            mask = torch.nan_to_num(stream_mask, nan=0.0).clamp(0.0, 1.0)
            logits = logits + (mask - 1.0) * 1e9

        # Temperature + mix with uniform prior
        τ = float(self.temp.clamp_min(1e-3))
        w = F.softmax(logits / τ, dim=1)
        m = float(self.mix.clamp(0.0, 1.0))
        uni = torch.full_like(w, 1.0 / w.size(1))
        w = (1.0 - m) * uni + m * w  # [B,N]

        # Optional top-k sparsification
        if self.topk_enabled and self.topk_cfg is not None and 1 <= self.topk_cfg < self.n_streams:
            topk_vals, topk_idx = torch.topk(w, self.topk_cfg, dim=1)
            keep = torch.zeros_like(w)
            keep.scatter_(1, topk_idx, 1.0)
            w = w * keep
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted concat of streams
        feats_out = []
        for i, f in enumerate(features):
            fi = self.ln_streams[i](f) if self.per_stream_norm and (self.ln_streams is not None) else f
            feats_out.append(fi * w[:, i:i+1])
        out = torch.cat(feats_out, dim=1)
        out = self.drop(out)

        # diagnostics + summary for head
        with torch.no_grad():
            ent = -(w * (w.clamp_min(1e-8)).log()).sum(dim=1)   # [B]
            maxw = w.max(dim=1).values                          # [B]
            conf_wmean = (w * sc).sum(dim=1, keepdim=True)      # [B,1]
            self.last_w = w.detach()
            self.last_logits = logits.detach()
            self.last_entropy = ent.mean()
            self.last_maxw = maxw.mean()

        if not return_aux:
            return out

        if self.per_stream_cond_dim is not None and cond_per_stream is not None:
            cond_flat = cond_per_stream.reshape(B, self.n_streams * self.per_stream_cond_dim)
        else:
            cond_flat = cond if cond is not None else torch.zeros(B, 0, device=out.device, dtype=out.dtype)

        cond_for_head = torch.cat([cond_flat,
                                   ent.unsqueeze(1),
                                   maxw.unsqueeze(1),
                                   conf_wmean], dim=1)

        aux = {
            "weights": w,
            "entropy": ent.mean(),
            "max_w": maxw.mean(),
            "logits": logits,
            "conf_used": sc,
            "cond_for_head": cond_for_head
        }
        return out, aux
