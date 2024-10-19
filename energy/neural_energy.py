from functools import cache

import torch
import torch.nn as nn
import numpy as np
import math

from .base_energy import BaseEnergy

class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int = 64):
        super(TimeEncoding, self).__init__()

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


class StateEncoding(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncoding, self).__init__()

        self.x_model = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim),
            nn.GELU(),
        )

    def forward(self, s):
        return self.x_model(s)

class NeuralEnergy(nn.Module): # Feedback
    def __init__(self, s_dim: int, harmonics_dim: int = 256, t_emb_dim: int = 256, hidden_dim: int = 256, s_emb_dim: int = 256, mlp_hidden_dim: int = 256):
        super(NeuralEnergy, self).__init__()
        
        self.state_encoder = StateEncoding(s_dim, hidden_dim, s_emb_dim)
        self.time_encoder = TimeEncoding(harmonics_dim, t_emb_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(s_emb_dim + t_emb_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
    
    def forward(self, x, t): # Feedback
        x_encoded = self.state_encoder(x)
        t_encoded = self.time_encoder(t)
        
        t_encoded = t_encoded.expand_as(x_encoded)
        
        combined = torch.cat([x_encoded, t_encoded], dim=-1)
        output = self.mlp(combined)
        
        return output.squeeze(-1)
    
    def score(self, x: torch.Tensor, t):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                (-self.forward(copy_x, t)).sum().backward()
                grad_energy = copy_x.grad.data
            return grad_energy