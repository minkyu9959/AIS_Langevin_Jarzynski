import torch
import numpy as np
import random

from energy.neural_drift import NeuralDrift

@torch.enable_grad()
def get_reward_and_gradient(x, log_reward):
    x = x.requires_grad_(True)
    log_r_x = log_reward(x)
    log_r_grad = torch.autograd.grad(log_r_x.sum(), x, create_graph=True)[0]

    return log_r_x, log_r_grad


def langevin_proposal(neural_drift, x, log_r_grad, step_size, sigma):
    
    dt = step_size #Minkyu
    transition_mean = (2*sigma*(neural_drift) + sigma**2 * log_r_grad) * dt #Fix_backwards
    transition_std = sigma * np.sqrt(2 * dt) #Fix_backwards
    
    x_new = ( #Fix_backwards
        x
        + transition_mean
        + transition_std * torch.randn_like(x, device=x.device)
    )
    
    return x_new, transition_mean, transition_std
    
def compute_transition_log_prob(x, x_new, transition_mean, transition_std): #Fix_backwards
    variance = torch.tensor(transition_std**2, device=x.device)
    
    # Compute the log probability of x_{t+1} under the Gaussian transition from x_t
    log_p_transition = (
        -((x_new - transition_mean) ** 2).sum(dim=-1) / (2 * variance) 
        - 0.5 * x.shape[1] * torch.log(2 * torch.pi * variance)
    ) 

    return log_p_transition


def one_step_langevin_dynamic(neural_drift, x, time, log_reward, step_size, sigma, do_correct=False):
    log_r_old, r_grad_old = get_reward_and_gradient(x, log_reward)
    drift_value = neural_drift(x, time)
    
    x_new, transition_mean, transition_std = langevin_proposal(drift_value, x, r_grad_old, step_size, sigma) #Fix_backwards
    
    log_transition_prob = compute_transition_log_prob(x, x_new, transition_mean, transition_std) #Fix_backwards

    return x_new, log_transition_prob
