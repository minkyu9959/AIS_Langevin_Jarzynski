import torch
from torch import Tensor

def reverse_KL_vae_reparam(log_weights: Tensor, target_log_Z: float, prior_log_Z: float) -> Tensor:
  return log_weights.mean() + target_log_Z - prior_log_Z

def reverse_KL_jarzynski_reparam(radon_nikodym: Tensor, target_log_Z: float, prior_log_Z: float) -> Tensor:
  return radon_nikodym.mean() + target_log_Z - prior_log_Z

def reverse_KL_jarzynski_reinforce(radon_nikodym: Tensor, log_p_X: Tensor, target_log_Z: float, prior_log_Z: float) -> Tensor:
  rn = radon_nikodym + target_log_Z - prior_log_Z
  rn_detach = rn.detach()
  return rn.mean() + (rn_detach * log_p_X).mean()

def vargrad(radon_nikodym: Tensor, target_log_Z: float, prior_log_Z: float) -> Tensor:
  rn = radon_nikodym + target_log_Z - prior_log_Z
  return (rn - rn.mean()).square().mean()

def tb_avg(TB_param: torch.nn.Parameter, radon_nikodym: torch.Tensor,target_log_Z: float, prior_log_Z: float) -> torch.Tensor:
  rn = radon_nikodym + target_log_Z - prior_log_Z
  return (rn - TB_param).square().mean()


def compute_loss(loss_type, log_weights, radon_nikodym, target_log_Z, prior_log_Z, TB_param, log_p_X) -> torch.Tensor:
  
  if loss_type == "KL_vae":
    return reverse_KL_vae_reparam(log_weights, target_log_Z, prior_log_Z)
  
  elif loss_type == "KL_jarzynski":
    return reverse_KL_jarzynski_reparam(radon_nikodym, target_log_Z, prior_log_Z)
  
  elif loss_type == "KL_jarzynski_reinforce":
    return reverse_KL_jarzynski_reinforce(radon_nikodym, log_p_X, target_log_Z, prior_log_Z)
  
  elif loss_type == "vargrad":
    return vargrad(radon_nikodym, target_log_Z, prior_log_Z)
  
  elif loss_type == "tb_avg":
    return tb_avg(TB_param, radon_nikodym, target_log_Z, prior_log_Z)
  
  else:
    raise ValueError(f"Unsupported loss type: {loss_type}")
