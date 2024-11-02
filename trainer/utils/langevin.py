import torch
import numpy as np
import random

from energy.neural_drift import NeuralDrift
from torch.distributions.normal import Normal

@torch.enable_grad()
def get_reward_and_gradient(x, log_reward):
    x = x.requires_grad_(True)
    log_r_x = log_reward(x)
    log_r_grad = torch.autograd.grad(log_r_x.sum(), x, create_graph=True)[0]

    return log_r_x, log_r_grad

def langevin_proposal(neural_drift, x, log_r_grad, step_size, sigma):
    
    dt = step_size 
    transition_mean = (2 * sigma * (neural_drift) + (sigma**2) * log_r_grad) * dt 
    transition_std = sigma * np.sqrt(2 * dt) 
    
    x_new = ( # Euler-Maruyama
        x
        + transition_mean
        + transition_std * torch.randn_like(x, device=x.device)
    )
    
    return x_new

def langevin_proposal_bwd(x, log_r_grad, step_size, sigma): # not used
        
    dt = step_size 
    transition_mean = - (sigma**2) * log_r_grad * dt 
    transition_std = sigma * np.sqrt(2 * dt) 
    
    x_prev = ( 
        x
        + transition_mean
        + transition_std * torch.randn_like(x, device=x.device)
    )
    
    return x_prev
    
def compute_transition_log_prob(x, x_new, neural_drift, log_r_grad, sigma, step_size): 
    
    dt = step_size
    
    mean_fwd = x + (2 * sigma * (neural_drift) + (sigma**2) * log_r_grad) * dt
    mean_bwd = x_new - (sigma**2) * log_r_grad * dt
    std = sigma * np.sqrt(2 * dt)
    
    gaussian_fwd = Normal(mean_fwd, std)
    gaussian_bwd = Normal(mean_bwd, std)

    log_p_transition_fwd = gaussian_fwd.log_prob(x_new).sum(dim=-1) # P(X_{t+1}|X_t) = N(X_{t+1}|X_t + fwd_drift, sigma^2)
    log_p_transition_bwd = gaussian_bwd.log_prob(x).sum(dim=-1) # P(X_t|X_{t+1}) = N(X_t|X_{t+1} + bwd_drift, sigma^2)
    
    log_importance_weight = log_p_transition_fwd - log_p_transition_bwd
    
    return log_p_transition_fwd, log_p_transition_bwd, log_importance_weight


def one_step_langevin_dynamic(neural_drift, x, time, log_reward, step_size, sigma, do_correct=False):
    log_r_old, r_grad_old = get_reward_and_gradient(x, log_reward)
    drift_value = neural_drift(x, time)
    
    x_new = langevin_proposal(drift_value, x, r_grad_old, step_size, sigma) # Next states
    
    # if trick_mode == "reinforce":
    #     x.detach()
    #     x_new = x_new.detach()
    
    log_p_transition_fwd, log_p_transition_bwd, log_importance_weight = compute_transition_log_prob(x, x_new, drift_value, r_grad_old, sigma, step_size) # Compute transition log prob

    return x_new, log_p_transition_fwd, log_p_transition_bwd, log_importance_weight
