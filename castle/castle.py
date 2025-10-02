# castle/castle.py
from __future__ import annotations
import torch
import torch.nn as nn

from .helper import LearnableTimeScale, MultiTimeScale
from .decay_aware import DecayAware
from .time_embedding import TimeEmbeddingGRUEncoder
from .tabular import TabularEncoder
from .stream_cond import StreamCondBuilder
from .fusion import FusionAllStream
from .anchor import AnchorCreator
from .volatility_analyser import VolatilityAnalyzer, ResidualReliability
from .regime import RegimeAdapter
from .head import ConsensusTrustHead

__all__ = ["CASTLE"]

class CASTLE(nn.Module):
    """
    Flow (strict):
      DecayAware → { AnchorCreator → VolatilityAnalyzer → RegimeAdapter ;
                     TE encoders + Tabular → StreamCondBuilder → Fusion }
      Head consumes ONLY {RegimeAdapter output, Fusion output}.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)

        # ---- dims ----
        Ds = int(self.cfg["soil_in"])
        Di = int(self.cfg["indoor_in"])
        Dw = int(self.cfg["weather_in"])
        Dc = int(self.cfg["core_dim"])                      # crop (daily)
        Da = int(self.cfg.get("anchor_dim", 5))             # anchors (5)
        K  = int(self.cfg.get("K", self.cfg.get("regime_K", self.cfg.get("k_hist", 6))))
        device = self.cfg.get("device", "cpu")

        # ---- time scales ----
        ts_init = float(self.cfg.get("ts_init_seconds", 3600.0))
        use_mts = bool(self.cfg.get("use_multi_timescale", True))
        self.mts_l2_pull = float(self.cfg.get("mts_l2_pull", 1e-3))

        if use_mts:
            # Per-stream τ plus a shared τ
            self.mts = MultiTimeScale(init_seconds=ts_init, l2_pull=self.mts_l2_pull)
            ts_shared  = self.mts.shared
            ts_soil    = self.mts.soil
            ts_indoor  = self.mts.indoor
            ts_weather = self.mts.weath
            ts_tab     = self.mts.tab
        else:
            # single shared τ
            self.ts_mod = LearnableTimeScale(init_seconds=ts_init, trainable=True)
            self.mts = None
            ts_shared = ts_soil = ts_indoor = ts_weather = ts_tab = self.ts_mod

        # ---- DecayAware (seq streams + daily streams) ----
        # DA is the single source of confidences; all encoders reuse them.
        self.da_soil    = DecayAware(input_size=Ds, time_scale_module=ts_soil,
                                     noise_scale=float(self.cfg.get("soil_noise", 0.01)))
        self.da_indoor  = DecayAware(input_size=Di, time_scale_module=ts_indoor,
                                     noise_scale=float(self.cfg.get("indoor_noise", 0.01)))
        self.da_weather = DecayAware(input_size=Dw, time_scale_module=ts_weather,
                                     noise_scale=float(self.cfg.get("weather_noise", 0.01)))
        self.da_crop    = DecayAware(input_size=Dc, time_scale_module=ts_tab, noise_scale=0.0)
        self.da_anchor  = DecayAware(input_size=Da, time_scale_module=ts_shared, noise_scale=0.0)

        # ---- Encoders ----
        te_hidden = int(self.cfg.get("soil_hidden", 64))
        te_out    = int(self.cfg.get("proj_dim", 32))
        self.te_soil = TimeEmbeddingGRUEncoder(
            input_dim=Ds, hidden_dim=te_hidden, output_dim=te_out,
            time_scale_module=ts_soil, use_internal_conf=False
        )
        self.te_indoor = TimeEmbeddingGRUEncoder(
            input_dim=Di, hidden_dim=te_hidden, output_dim=te_out,
            time_scale_module=ts_indoor, use_internal_conf=False
        )
        self.te_weather = TimeEmbeddingGRUEncoder(
            input_dim=Dw, hidden_dim=te_hidden, output_dim=te_out,
            time_scale_module=ts_weather, use_internal_conf=False
        )

        tab_hidden = int(self.cfg.get("tab_hidden", 64))
        tab_out    = int(self.cfg.get("proj_dim", 32))
        self.tabular = TabularEncoder(
            input_dim=Dc, hidden_dim=tab_hidden, output_dim=tab_out,
            use_mask=False, use_internal_conf=False,
            dropout=float(self.cfg.get("tab_dropout", 0.05))
        )

        # ---- Drift-control trio ----
        self.anchor_creator = AnchorCreator(
            temperature=float(self.cfg.get("anchor_temperature", 1.3)),
            shock_gain=float(self.cfg.get("anchor_shock_gain", 1.0)),
            tau_init=float(self.cfg.get("anchor_tau_init", 3.0)),
            tau_min=float(self.cfg.get("anchor_tau_min", 1e-3)),
            tau_max=float(self.cfg.get("anchor_tau_max", 1e3)),
        )
        self.vol_analyzer = VolatilityAnalyzer(
            vol_dim=int(self.cfg.get("vol_dim", 8)),
            z_clip=float(self.cfg.get("vol_z_clip", 6.0)),
            eps=float(self.cfg.get("vol_eps", 1e-8)),
            hidden=int(self.cfg.get("vol_hidden", 32)),
            dropout=float(self.cfg.get("vol_dropout", 0.05)),
            rel_hidden=int(self.cfg.get("vol_rel_hidden", 32)),
            rel_min_conf=float(self.cfg.get("vol_rel_min_conf", 1e-3)),
            leash_blend=float(self.cfg.get("vol_leash_blend", 0.30)),
        )
        self.regime = RegimeAdapter(
            cond_dim=20,
            K=int(self.cfg.get("regime_K", K)),
            a_span=float(self.cfg.get("regime_a_span", 0.25)),
            lambda_max=float(self.cfg.get("regime_lambda_max", 0.80)),
            lambda_init=float(self.cfg.get("regime_lambda_init", 0.20)),
            gate_temp=float(self.cfg.get("regime_gate_temp", 1.0)),
            gate_entropy_reg=float(self.cfg.get("regime_gate_entropy_reg", 0.0)),
            use_cond_norm=bool(self.cfg.get("regime_use_cond_norm", True)),
        )

        # ---- Per-stream conditions & Fusion ----
        # Use the shared τ for Δ stats in StreamCondBuilder
        self.cond_builder = StreamCondBuilder(time_scale_module=ts_shared)
        self.seqstats_last_k = int(self.cfg.get("seqstats_last_k", 12))
        self.per_stream_cond_dim = 5

        self.fusion = FusionAllStream(
            input_dims=[te_out, te_out, te_out, tab_out],
            fusion_dim=int(self.cfg.get("fusion_proj", 64)),
            per_stream_cond_dim=self.per_stream_cond_dim,
            init_mix=float(self.cfg.get("fusion_mix", 0.65)),
            init_temp=float(self.cfg.get("fusion_temp", 1.0)),
            per_stream_norm=True,
            topk=self.cfg.get("fusion_topk", None),
            dropout=float(self.cfg.get("fusion_dropout", 0.05)),
            gain_max=float(self.cfg.get("fusion_gain_max", 0.20)),
        )

        # ---- Head ----
        self.head = ConsensusTrustHead(
            fused_dim=te_out*3 + tab_out,
            base_hidden=int(self.cfg.get("head_hidden", 64)),
            delta_clip=float(self.cfg.get("head_delta_clip", 6.0)),
            dropout=float(self.cfg.get("head_dropout", 0.10)),
            use_hetero=bool(self.cfg.get("head_use_hetero", False)),
            tau_core=float(self.cfg.get("head_tau_core", 0.90)),
            tau_reg=float(self.cfg.get("head_tau_reg", 0.90)),
            reg_gain_max=float(self.cfg.get("head_reg_gain_max", 0.15)),
            reg_bias_max=float(self.cfg.get("head_reg_bias_max", 0.50)),
            core_inflate=float(self.cfg.get("head_core_inflate", 0.60)),
            reg_inflate=float(self.cfg.get("head_reg_inflate", 0.60)),
            core_conf_gain=float(self.cfg.get("head_core_conf_gain", 0.50)),
            reg_shock_gain=float(self.cfg.get("head_reg_shock_gain", 0.50)),
        )

        # ---- runtime toggles ----
        self.enable_topk_after   = int(self.cfg.get("enable_topk_after", -1))
        self.enable_memory_after = int(self.cfg.get("enable_memory_after", -1))

        # ---- Residual reliability ----
        self.use_resid_rel = bool(self.cfg.get("use_resid_rel", True))
        self.resid_rel = ResidualReliability(
            k=int(self.cfg.get("resid_rel_k", 7)),
            gain=float(self.cfg.get("resid_rel_gain", 0.35)),
            clip=float(self.cfg.get("resid_rel_clip", 6.0)),
        )
        self._res_fifo: torch.Tensor | None = None  # [B,k]

        self.to(device)

    # ---------------- helpers ----------------
    def _run_da_daily(self, x, m, d, da: "DecayAware"):
        """ x,m,d: [B,D] → add T=1 → DA.forward() → squeeze """
        x1, m1, d1 = x.unsqueeze(1), m.unsqueeze(1), d.unsqueeze(1)
        xhat_seq, conf_seq = da.forward(x1, m1, d1)
        return xhat_seq.squeeze(1), conf_seq.squeeze(1)  # [B,D], [B,1]

    def maybe_update_runtime(self, step: int):
        if (self.enable_topk_after >= 0) and (step >= self.enable_topk_after):
            self.fusion.enable_topk(self.cfg.get("fusion_topk", None))
        if (self.enable_memory_after >= 0) and (step >= self.enable_memory_after):
            if hasattr(self.regime, "enable_memory"):
                self.regime.enable_memory()

    # ---------------- residual FIFO (RR) ----------------
    def init_residual_buffer(self, batch_size: int, device=None, dtype=None):
        if not self.use_resid_rel:
            self._res_fifo = None
            return
        k = int(self.resid_rel.k)
        device = device or next(self.parameters()).device
        dtype  = dtype  or next(self.parameters()).dtype
        self._res_fifo = torch.zeros(batch_size, k, device=device, dtype=dtype)

    @torch.no_grad()
    def update_residuals(self, abs_resid: torch.Tensor):
        """Call from your loop in causal order: abs_resid: [B,1] (or [B])"""
        if (not self.use_resid_rel):
            return
        if abs_resid.dim() == 1:
            abs_resid = abs_resid.view(-1, 1)
        if self._res_fifo is None:
            self.init_residual_buffer(abs_resid.size(0), device=abs_resid.device, dtype=abs_resid.dtype)
        r = abs_resid.detach().reshape(abs_resid.size(0), 1).to(self._res_fifo.device, self._res_fifo.dtype)
        self._res_fifo = torch.cat([self._res_fifo[:, 1:], r], dim=1)

    def clear_residual_buffer(self):
        self._res_fifo = None

    # ---------------- optional regularizer ----------------
    def timescale_regularizer(self):
        """Small pull to keep per-stream τ close to shared τ (only if MultiTimeScale is used)."""
        if self.mts is None:
            return torch.zeros([], device=next(self.parameters()).device)
        return self.mts.penalty()

    # ---------------- forward (flow preserved) ----------------
    def forward(
        self,
        *,
        soil_x, soil_m, soil_d,
        indoor_x, indoor_m, indoor_d,
        weather_x, weather_m, weather_d,
        crop_x, crop_m, crop_d,
        anchor_x, anchor_m, anchor_d
    ):
        # 1) DecayAware (single source of confidences)
        soil_xh, soil_c = self.da_soil.forward(soil_x,    soil_m,    soil_d)
        ind_xh,  ind_c  = self.da_indoor.forward(indoor_x, indoor_m, indoor_d)
        wea_xh,  wea_c  = self.da_weather.forward(weather_x, weather_m, weather_d)

        crop_xh,  crop_c   = self._run_da_daily(crop_x,  crop_m,  crop_d,  self.da_crop)
        anchor_xh, _       = self._run_da_daily(anchor_x, anchor_m, anchor_d, self.da_anchor)  # [B,5]
        assert anchor_xh.dim() == 2 and anchor_xh.size(1) == 5, "Anchor must be [B,5]"

        # 2) Drift-control lane
        baseline, feat, mix, conf_vec, conf_glob, shock, summary, ac_dbg = self.anchor_creator(anchor_xh)
        v_emb, r_vol, va_diag = self.vol_analyzer(anchor_xh, summary=summary)

        # Optional residual reliability blend (strictly past)
        r_final = r_vol
        if self.use_resid_rel:
            if (self._res_fifo is None) or (self._res_fifo.size(0) != baseline.size(0)):
                self.init_residual_buffer(baseline.size(0), device=baseline.device, dtype=baseline.dtype)
            if self._res_fifo is not None:
                r_resid = self.resid_rel(self._res_fifo.to(baseline.device, baseline.dtype),
                                         scale_hint=baseline.abs())
                r_final = self.resid_rel.blend(r_vol, r_resid)

        ra_out = self.regime(
            y_core=baseline,
            summary=summary,
            r=r_final,
            update_memory=self.training,
            return_controls=True
        )
        y_regime = ra_out["y_cal"]
        r_regime = ra_out["reliability"]

        # 3) Feature lane
        soil_feat, _ = self.te_soil(soil_xh,  soil_m,   soil_d,  conf_override=soil_c)
        ind_feat,  _ = self.te_indoor(ind_xh, indoor_m, indoor_d, conf_override=ind_c)
        wea_feat,  _ = self.te_weather(wea_xh, weather_m, weather_d, conf_override=wea_c)
        tab_feat     = self.tabular(crop_xh, mask=crop_m, conf_override=crop_c)

        k = int(self.seqstats_last_k)
        cond_soil = self.cond_builder(soil_xh, soil_m, soil_d, soil_c, last_k=k)
        cond_ind  = self.cond_builder(ind_xh,  indoor_m, indoor_d, ind_c,  last_k=k)
        cond_wea  = self.cond_builder(wea_xh,  weather_m, weather_d, wea_c, last_k=k)
        cond_tab  = self.cond_builder(crop_xh.unsqueeze(1),
                                      crop_m.unsqueeze(1),
                                      crop_d.unsqueeze(1),
                                      crop_c.unsqueeze(1),
                                      last_k=1)
        cond_ps = torch.stack([cond_soil, cond_ind, cond_wea, cond_tab], dim=1)

        conf_soil = soil_c.mean(dim=1)
        conf_ind  = ind_c.mean(dim=1)
        conf_wea  = wea_c.mean(dim=1)
        conf_tab  = crop_c
        stream_conf = torch.cat([conf_soil, conf_ind, conf_wea, conf_tab], dim=1)

        fused, aux_fuse = self.fusion(
            features=[soil_feat, ind_feat, wea_feat, tab_feat],
            cond_per_stream=cond_ps,
            stream_conf=stream_conf,
            stream_mask=None,
            return_aux=True
        )

        # Training-only hook for regularizers (e.g., Jacobian external)
        fused_for_reg = fused if (self.training and bool(self.cfg.get("return_fused_for_reg", True))) else None

        conf_fuse  = aux_fuse["cond_for_head"][:, -1:]  # [B,1]
        scale_hint = baseline
        shock_b    = shock

        # 4) Head (consensus)
        y_pred, aux_head = self.head(
            fused=fused,
            regime_y=y_regime,
            regime_r=r_regime,
            shock=shock_b,
            conf_fuse=conf_fuse,
            scale_hint=scale_hint,
            return_alpha=True
        )

        # 5) Diagnostics
        aux = {
            "fusion_weights": aux_fuse["weights"],
            "fusion_entropy": aux_fuse["entropy"],
            "fusion_maxw": aux_fuse["max_w"],
            "fusion_conf_wmean": conf_fuse,
            "baseline": baseline,
            "ac_feat": feat,
            "ac_mix": mix,
            "ac_conf_vec": conf_vec,
            "ac_conf_glob": conf_glob,
            "shock": shock_b,
            "summary": summary,
            "vol_emb": v_emb,
            "vol_diag": va_diag,
            "y_regime": y_regime,
            "r_regime": r_regime,
            "r_final_used": r_final,
            "regime_ctrl": {k: ra_out[k] for k in ("gate_w","a","lambda","leash_scale","adapt_gain","stability","router_entropy","aux_reg","last_reg") if k in ra_out},
            "alpha": aux_head["alpha"],
            "y_core": aux_head["y_core"],
            "y_final": aux_head["y_final"],
            "delta": aux_head["delta"],
            "delta_norm": aux_head["delta_norm"],
            "factors": {
                "w_core": aux_head["w_core"],
                "w_reg":  aux_head["w_reg"],
                "logv_core": aux_head["logv_core"],
                "logv_reg":  aux_head["logv_reg"],
                "reg_calib_gain": aux_head["reg_calib"]["gain"],
                "reg_calib_bias": aux_head["reg_calib"]["bias"],
            },
            "fused_for_reg": fused_for_reg,
        }
        return y_pred, aux
