import torch
import torch.nn as nn
import torch.nn.functional as F
from monotone_decay import MonotoneDecay

class DecayAware(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, device="cpu", time_scale: float = 3600.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.time_scale = time_scale

        # Embed AFTER robust imputation
        self.input_embed = nn.Linear(input_size, input_size)

        # Monotone decays (D -> H) and (D -> D)
        self.decay_x = MonotoneDecay(input_size, input_size)
        self.decay_h = MonotoneDecay(input_size, hidden_size)

        # Confidence gate takes [x_hat, d_norm, m_t] -> per-feature confidence
        self.confidence_layer = nn.Linear(input_size * 3, input_size)

        # GRU-like gates (inputs = x_final_emb, m_t, d_norm, h)
        gate_in = input_size * 3 + hidden_size
        self.z_gate = nn.Linear(gate_in, hidden_size)
        self.r_gate = nn.Linear(gate_in, hidden_size)
        self.h_tilde = nn.Linear(gate_in, hidden_size)

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size) if output_size else nn.Identity(),
        )

        # Global mean buffer in RAW feature space (avoids future leakage)
        self.register_buffer("x_global_mean", torch.zeros(1, 1, input_size))

    @torch.no_grad()
    def set_global_mean(self, mean_vec):  # mean_vec: [D] in RAW space
        self.x_global_mean = mean_vec.view(1, 1, -1)

    def _norm_delta(self, d):
        # clamp and normalize consistently
        d = torch.clamp(torch.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0), 0.0)
        return torch.log1p(d / self.time_scale)

    def forward(self, x, x_mask, x_delta, x_mean=None):
        """
        x:       [B, T, D] RAW features (may contain NaN/Inf)
        x_mask:  [B, T, D] 1 if observed else 0
        x_delta: [B, T, D] time since last obs per feature (same units across sites if possible)
        x_mean:  [B, 1, D] optional causal/site mean in RAW space
        """
        B, T, D = x.shape

        # Work in RAW space for imputation logic
        x_raw = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x_mean is None:
            x_mean = self.x_global_mean.expand(B, 1, D)
        x_mean = torch.nan_to_num(x_mean, nan=0.0, posinf=0.0, neginf=0.0)

        h = torch.zeros(B, self.hidden_size, device=self.device)
        outputs = []

        # Track last observed RAW values per feature (init with mean)
        x_last = x_mean.squeeze(1).clone()

        for t in range(T):
            x_t = x_raw[:, t, :]
            m_t = torch.nan_to_num(x_mask[:, t, :], nan=0.0, posinf=0.0, neginf=0.0)
            d_t = self._norm_delta(x_delta[:, t, :])  # normalized deltas

            # Learned, monotone decays
            gamma_h = self.decay_h(d_t)  # (B, H) in (0,1]
            gamma_x = self.decay_x(d_t)  # (B, D) in (0,1]

            # Hidden-state decay
            h = gamma_h * h

            # Impute in RAW space
            x_fallback = gamma_x * x_last + (1.0 - gamma_x) * x_mean.squeeze(1)
            x_hat = m_t * x_t + (1.0 - m_t) * x_fallback

            # Confidence that x_hat is reliable; never attenuate observed features
            conf_logits = self.confidence_layer(torch.cat([x_hat, d_t, m_t], dim=1))
            conf = 0.2 + 0.8 * torch.sigmoid(conf_logits)  # floor at 0.2 to avoid collapse
            conf = torch.where(m_t.bool(), torch.ones_like(conf), conf)

            # Soft blend between imputed value and its fallback (no shrink-to-zero at eval)
            x_final_raw = conf * x_hat + (1.0 - conf) * x_fallback

            if self.training:
                noise = torch.randn_like(x_final_raw)
                x_final_raw = x_final_raw + (1.0 - conf) * noise

            # Now embed ONCE after robust imputation
            x_final = self.input_embed(x_final_raw)

            inputs = torch.cat([x_final, m_t, d_t, h], dim=1)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)

            z = torch.sigmoid(self.z_gate(inputs))
            r = torch.sigmoid(self.r_gate(inputs))
            h_tilde = torch.tanh(self.h_tilde(torch.cat([x_final, m_t, d_t, r * h], dim=1)))
            h = (1.0 - z) * h + z * h_tilde
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

            # Update last observed RAW values
            x_last = torch.where(m_t.bool(), x_t, x_last)
            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [B, T, H]
        return self.output(torch.nan_to_num(outputs[:, -1, :], nan=0.0, posinf=0.0, neginf=0.0))
