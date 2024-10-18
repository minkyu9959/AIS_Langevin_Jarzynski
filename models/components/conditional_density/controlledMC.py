from typing import Callable

import torch
from torch import Tensor

import numpy as np

from .interface import ConditionalDensity
from ..architectures import *


class ControlledMCConditional(ConditionalDensity):
    """
    Conditional density family using annealed score with control.
    """

    def __init__(
        self,
        dt: float,
        sample_dim: int,
        state_encoder: torch.nn.Module,
        time_encoder: torch.nn.Module,
        control_model: torch.nn.Module,
        do_control_plus_score: bool,
        annealed_score_fn: Callable,
        base_std: float = 1.0,
        clipping: bool = False,
        lgv_clip: float = 1e2,
        gfn_clip: float = 1e4,
    ):
        self.do_control_plus_score = do_control_plus_score

        self.sample_dim = sample_dim

        self.dt = dt

        self.state_encoder = state_encoder
        self.time_encoder = time_encoder
        self.control_model = control_model

        self.annealed_score_fn = annealed_score_fn
        self.base_std = base_std

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

    def get_score_and_control(self, state: Tensor, time: float):
        assert state.dim() == 2 and state.size(1) == self.sample_dim

        batch_size = state.shape[0]

        encoded_time = self.time_encoder(time).repeat(batch_size, 1)
        encoded_state = self.state_encoder(state)

        mean_and_logvar = self.control_model(encoded_state, encoded_time)

        # Get annealed score and clip it.
        annealed_score = self.annealed_score_fn(time, state)
        if self.clipping:
            annealed_score = torch.clip(annealed_score, -self.lgv_clip, self.lgv_clip)

        return mean_and_logvar, annealed_score

    def params(self, state: Tensor, time: float) -> dict:
        mean_and_logvar, annealed_score = self.get_score_and_control(state, time)

        if self.do_control_plus_score:
            mean_and_logvar[..., : self.sample_dim] += ((self.base_std**2) / 2) * annealed_score
        else:
            mean_and_logvar[..., : self.sample_dim] = ((self.base_std**2) / 2) * annealed_score - mean_and_logvar[..., : self.sample_dim]

        if self.clipping:
            mean_and_logvar = torch.clip(mean_and_logvar, -self.gfn_clip, self.gfn_clip)

        mean, logvar = ConditionalDensity.split_gaussian_params(mean_and_logvar)

        # mask the model output and set the variance as constant.
        logvar = torch.full_like(logvar, np.log(self.base_std) * 2.0 + np.log(2))

        mean = mean * self.dt
        mean = mean + state

        logvar = logvar + np.log(self.dt)

        return {"mean": mean, "logvar": logvar}
