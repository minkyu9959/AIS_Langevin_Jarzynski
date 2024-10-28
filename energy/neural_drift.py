from functools import cache

import torch
import torch.nn as nn
import numpy as np
import math

class TimeEmbedding(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int = 64):
        super(TimeEmbedding, self).__init__()

        pe = torch.arange(1, harmonics_dim + 1).float().unsqueeze(0) * 2 * math.pi
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
            nn.GELU(),
        )
        self.register_buffer("pe", pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = (t * self.pe).sin()
        t_cos = (t * self.pe).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEmbedding(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEmbedding, self).__init__()

        self.x_model = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim),
            nn.GELU(),
        )

    def forward(self, s):
        return self.x_model(s)

class NeuralDrift(nn.Module): # Feedback
    def __init__(self, s_dim: int, harmonics_dim: int = 128, t_emb_dim: int = 128, hidden_dim: int = 128, s_emb_dim: int = 128, mlp_hidden_dim: int = 128):
        super(NeuralDrift, self).__init__()
        
        self.state_embedder = StateEmbedding(s_dim, hidden_dim, s_emb_dim)
        self.time_embedder = TimeEmbedding(harmonics_dim, t_emb_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(s_emb_dim + t_emb_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, s_dim)
        )
    
    def forward(self, x, t): # Feedback
          
        x_embedded = self.state_embedder(x)
        t_embedded = self.time_embedder(t)
        
        t_embedded = t_embedded.expand(x_embedded.size(0), -1)
        
        combined = torch.cat([x_embedded, t_embedded], dim=-1)
        output = self.mlp(combined)
        
        return output.squeeze(-1)