import torch
import numpy as np


def adjust_ld_step(
    current_ld_step,
    current_acceptance_rate,
    target_acceptance_rate=0.574,
    adjustment_factor=0.01,
):
    """
    Adjust the Langevin dynamics step size based on the current acceptance rate.

    :param current_ld_step: Current Langevin dynamics step size.
    :param current_acceptance_rate: Current observed acceptance rate.
    :param target_acceptance_rate: Target acceptance rate, default is 0.574.
    :param adjustment_factor: Factor to adjust the ld_step.
    :return: Adjusted Langevin dynamics step size.
    """
    if current_acceptance_rate > target_acceptance_rate:
        return current_ld_step + adjustment_factor * current_ld_step
    else:
        return current_ld_step - adjustment_factor * current_ld_step


def langevin_dynamics(x, log_reward, device, cfg):
    accepted_samples = []
    accepted_logr = []
    acceptance_rate_lst = []
    log_r_original = log_reward(x)
    acceptance_count = 0
    acceptance_rate = 0
    total_proposals = 0

    for i in range(cfg.max_iter_ls):
        x = x.requires_grad_(True)

        r_grad_original = torch.autograd.grad(log_reward(x).sum(), x)[0]
        if cfg.ld_schedule:
            ld_step = (
                cfg.ld_step
                if i == 0
                else adjust_ld_step(
                    ld_step,
                    acceptance_rate,
                    target_acceptance_rate=cfg.target_acceptance_rate,
                )
            )
        else:
            ld_step = cfg.ld_step

        new_x = (
            x
            + ld_step * r_grad_original.detach()
            + np.sqrt(2 * ld_step) * torch.randn_like(x, device=device)
        )
        log_r_new = log_reward(new_x)
        r_grad_new = torch.autograd.grad(log_reward(new_x).sum(), new_x)[0]

        log_q_fwd = -(
            torch.norm(new_x - x - ld_step * r_grad_original, p=2, dim=1) ** 2
        ) / (4 * ld_step)
        log_q_bck = -(torch.norm(x - new_x - ld_step * r_grad_new, p=2, dim=1) ** 2) / (
            4 * ld_step
        )

        log_accept = (log_r_new - log_r_original) + log_q_bck - log_q_fwd
        accept_mask = torch.rand(x.shape[0], device=device) < torch.exp(
            torch.clamp(log_accept, max=0)
        )
        acceptance_count += accept_mask.sum().item()
        total_proposals += x.shape[0]

        x = x.detach()
        # After burn-in process
        if i > cfg.burn_in:
            accepted_samples.append(new_x[accept_mask])
            accepted_logr.append(log_r_new[accept_mask])
        x[accept_mask] = new_x[accept_mask]
        log_r_original[accept_mask] = log_r_new[accept_mask]

        if i % 5 == 0:
            acceptance_rate = acceptance_count / total_proposals
            if i > cfg.burn_in:
                acceptance_rate_lst.append(acceptance_rate)
            acceptance_count = 0
            total_proposals = 0

    return torch.cat(accepted_samples, dim=0), torch.cat(accepted_logr, dim=0)


@torch.enable_grad()
def get_reward_and_gradient(x, log_reward):
    x = x.requires_grad_(True)
    log_r_x = log_reward(x)
    log_r_grad = torch.autograd.grad(log_r_x.sum(), x)[0]

    return log_r_x, log_r_grad


def langevin_proposal(x, log_r_grad, step_size):
    return (
        x
        + step_size * log_r_grad.detach()
        + np.sqrt(2 * step_size) * torch.randn_like(x, device=x.device)
    ).detach()


def correction_step(
    old_x, log_r_old, r_grad_old, new_x, log_r_new, r_grad_new, step_size
):
    device = old_x.device

    log_q_fwd = -(torch.norm(-old_x - step_size * r_grad_old, p=2, dim=1) ** 2) / (
        4 * step_size
    )

    log_q_bck = -(
        torch.norm(old_x - new_x - step_size * r_grad_new, p=2, dim=1) ** 2
    ) / (4 * step_size)

    log_accept = (log_r_new - log_r_old) + log_q_bck - log_q_fwd
    accept_mask = torch.rand(old_x.shape[0], device=device) < torch.exp(
        torch.clamp(log_accept, max=0)
    )

    return accept_mask


def one_step_langevin_dynamic(x, log_reward, step_size, do_correct=False):
    log_r_old, r_grad_old = get_reward_and_gradient(x, log_reward)

    new_x = langevin_proposal(x, r_grad_old, step_size)

    if do_correct:
        log_r_new, r_grad_new = get_reward_and_gradient(new_x, log_reward)
        accept_mask = correction_step(
            x, log_r_old, r_grad_old, new_x, log_r_new, r_grad_new, step_size
        )
        x_c = x.clone()  # x를 복제하여 x_c에 할당
        x_c[accept_mask] = new_x[accept_mask]
        x = x_c  # x_c를 x에 다시 할당
        
        #x[accept_mask] = new_x[accept_mask]
    else:
        x = new_x

    return x.detach()
