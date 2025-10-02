import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModelAttentionFusion(nn.Module):
    def __init__(self, input_dims, fusion_dim):
        super().__init__()
        self.attn_proj = nn.Linear(sum(input_dims), fusion_dim)
        self.attn_weights = nn.Linear(fusion_dim, len(input_dims))

    def forward(self, features):
        features = [torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0) for f in features]
        all_feat = torch.cat(features, dim=1)
        attn_scores = torch.tanh(self.attn_proj(all_feat))
        weights = F.softmax(self.attn_weights(attn_scores), dim=1)
        return torch.cat([f * weights[:, i:i + 1] for i, f in enumerate(features)], dim=1)
