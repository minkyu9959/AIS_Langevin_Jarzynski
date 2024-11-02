import torch
from torch import Tensor
from omegaconf import DictConfig

def compute_elbo_vae_loss(log_weights: Tensor, target_log_Z: float, prior_log_Z: float) -> Tensor:
  return log_weights.mean() + target_log_Z

def compute_elbo_jarzynski_loss(radon_nikodym: Tensor, target_log_Z: float, prior_log_Z: float) -> Tensor:
  return radon_nikodym.mean()

def compute_vargrad_loss(radon_nikodym: Tensor, target_log_Z: float, prior_log_Z: float) -> Tensor:
  return (radon_nikodym - radon_nikodym.mean()).square().mean()

def compute_tb_avg_loss(TB_param: torch.nn.Parameter, radon_nikodym: torch.Tensor, target_log_Z: float, prior_log_Z: float) -> torch.Tensor:
  return (radon_nikodym - TB_param).square().mean()

def compute_loss(loss_type, log_weights, radon_nikodym, target_log_Z, prior_log_Z, TB_param) -> torch.Tensor:
  if loss_type == "elbo_vae":
    return compute_elbo_vae_loss(log_weights, target_log_Z, prior_log_Z)
  elif loss_type == "elbo_jarzynski":
    return compute_elbo_jarzynski_loss(radon_nikodym, target_log_Z, prior_log_Z)
  elif loss_type == "vargrad":
    return compute_vargrad_loss(radon_nikodym, target_log_Z, prior_log_Z)
  elif loss_type == "tb_avg":
    return compute_tb_avg_loss(TB_param, radon_nikodym, target_log_Z, prior_log_Z)
  else:
    raise ValueError(f"Unsupported loss type: {loss_type}")
