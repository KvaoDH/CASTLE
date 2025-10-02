import torch
import torch.nn as nn
from feature_norm import FeatureNorm
from decay_aware import DecayAware
from time_embedding_gru_encoder import TimeEmbeddingGRUEncoder
from tabular_encoder import TabularEncoder
from cross_model_attention_fusion import CrossModelAttentionFusion

class HybridAgriModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Branch encoders
        self.soil_encoder = DecayAware(
            config['soil_in'], config['soil_hidden'], config['branch_out'],
            device=config.get('device', 'cpu'), time_scale=config.get('time_scale', 3600.0)
        )

        env_in = config['indoor_in'] + config['weather_in']
        self.env_norm = FeatureNorm(env_in)
        self.env_encoder = TimeEmbeddingGRUEncoder(
            env_in, 32, config['branch_out'], time_scale=config.get('time_scale', 3600.0),
            include_mask=True, include_delta=True
        )

        self.crop_norm = FeatureNorm(config['crop_in'])
        self.crop_encoder = TabularEncoder(config['crop_in'], 16, config['branch_out'], use_mask=True)

        # Fusion + head
        self.fusion = CrossModelAttentionFusion([config['branch_out']] * 3, config['fusion_dim'])

        self.head = nn.Sequential(
            nn.LayerNorm(config['branch_out'] * 3),
            nn.Dropout(0.3),
            nn.Linear(config['branch_out'] * 3, 64),
            nn.ReLU(),
            nn.Linear(64, config['output_dim']),
            nn.Identity() if not config.get('nonneg', True) else nn.Softplus(),
        )

    def forward(self, soil, mask, delta, indoor, weather, crop,
                env_mask=None, env_delta=None, crop_mask=None, x_mean=None):
        # Soil branch (supports per-sequence/site mean via x_mean)
        f1 = self.soil_encoder(soil, mask, delta, x_mean=x_mean)

        # Env branch: use true env deltas & masks if provided, else infer
        env_x = torch.cat([indoor, weather], dim=2)
        if env_mask is None:
            env_mask = torch.isfinite(env_x).float()
        if env_delta is None:
            # Fallback: scalar step deltas from soil; better: pass real env deltas
            env_delta = delta.mean(dim=2)
        env_x = self.env_norm(env_x)
        f2 = self.env_encoder(env_x, env_delta, mask=env_mask)

        # Crop branch: include mask
        if crop_mask is None:
            crop_mask = torch.isfinite(crop).float()
        crop_n = self.crop_norm(crop)
        f3 = self.crop_encoder(crop_n, mask=crop_mask)

        return self.head(self.fusion([f1, f2, f3]))
