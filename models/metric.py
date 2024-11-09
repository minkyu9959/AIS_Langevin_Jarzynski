import torch
from torch import Tensor

def lower_bound_vae(log_weights: Tensor, prior_log_Z: float) -> Tensor:  
  return -log_weights.mean() + prior_log_Z

def lower_bound_jarzynski(radon_nikodym: Tensor, prior_log_Z: float) -> Tensor:
  return -radon_nikodym.mean() + prior_log_Z


def compute_lower_bound(loss_type, log_weights, radon_nikodym, prior_log_Z) -> torch.Tensor:
    
    if loss_type == "KL_vae":
        return lower_bound_vae(log_weights, prior_log_Z)
    
    else:
        return lower_bound_jarzynski(radon_nikodym, prior_log_Z)