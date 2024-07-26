from typing import Callable, Optional
from functools import partial

import torch

import numpy as np

from models import GFN
from buffer import *
from energy import BaseEnergy

from .gfn_loss import *
from omegaconf import DictConfig


def calculate_subtb_coeff_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def get_buffer(
    buffer_cfg: Optional[DictConfig], energy_function: BaseEnergy
) -> Optional[BaseBuffer]:

    if buffer_cfg is None:
        return None

    if buffer_cfg.prioritized:
        buffer_class = PrioritizedReplayBuffer
    else:
        buffer_class = SimpleReplayBuffer

    return buffer_class(
        buffer_size=buffer_cfg.buffer_size,
        device=buffer_cfg.device,
        log_reward_function=energy_function.log_reward,
        batch_size=buffer_cfg.batch_size,
        data_ndim=energy_function.data_ndim,
        beta=buffer_cfg.beta,
    )


def get_gfn_forward_loss(
    loss_type: str,
    coeff_matrix: Optional[torch.Tensor] = None,
):
    """
    Get forward loss function based on the loss type.
    Returned loss functions have common interface.
    loss_fn(initial_state, gfn, log_reward_fn, exploration_std=None, return_exp=False)
    """
    if loss_type == "tb":
        return GFNForwardLossWrapper(trajectory_balance_loss)

    elif loss_type == "tb-avg":
        return GFNForwardLossWrapper(vargrad_loss)

    elif loss_type == "db":
        return GFNForwardLossWrapper(detailed_balance_loss)

    elif loss_type == "pis":
        return pis

    else:
        return Exception("Invalid forward loss type")


def get_gfn_backward_loss(loss_type: str) -> Callable:
    if loss_type == "tb":
        return GFNBackwardLossWrapper(trajectory_balance_loss)

    elif loss_type == "tb-avg":
        return GFNBackwardLossWrapper(vargrad_loss)

    elif loss_type == "mle":
        return GFNBackwardLossWrapper(mle_loss)

    else:
        raise Exception("Invalid backward loss type")


# def get_gfn_forward_loss(
#     loss_type: str,
#     coeff_matrix: Optional[torch.Tensor] = None,
# ):
#     """
#     Get forward loss function based on the loss type.
#     Returned loss functions have common interface.
#     loss_fn(initial_state, gfn, log_reward_fn, exploration_std=None, return_exp=False)
#     """
#     if loss_type == "tb":
#         return fwd_tb

#     elif loss_type == "tb-avg":
#         return fwd_tb_avg

#     elif loss_type == "db":
#         return db

#     elif loss_type == "subtb":
#         subtb_loss_fn = partial(coeff_matrix=coeff_matrix)
#         return subtb_loss_fn

#     elif loss_type == "pis":
#         return pis

#     else:
#         return Exception("Invalid forward loss type")


# def get_gfn_backward_loss(loss_type: str) -> Callable:
#     if loss_type == "tb":
#         return bwd_tb

#     elif loss_type == "tb-avg":
#         return bwd_tb_avg

#     elif loss_type == "mle":
#         return bwd_mle

#     else:
#         raise Exception("Invalid backward loss type")


def get_exploration_std(
    epoch, exploratory, exploration_factor=0.1, exploration_wd=False
):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1.0 - epoch / 5000.0)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl
