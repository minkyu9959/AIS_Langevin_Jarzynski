import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from omegaconf import DictConfig

from tqdm import tqdm
from argparse import ArgumentParser

from trainer.utils.langevin import one_step_langevin_dynamic
from energy import (
    AnnealedDensities,
    AnnealedEnergy,
    BaseEnergy,
    get_energy_function,
    GaussianEnergy,
)
from utility import SamplePlotter

#@torch.no_grad()
def annealed_IS_with_langevin(prior: BaseEnergy, target: BaseEnergy, cfg: DictConfig):
    annealed_densities = AnnealedDensities(target, prior)

    device = cfg.device
    num_time_steps = cfg.num_time_steps
    num_samples = cfg.num_samples

    sample = prior.sample(num_samples, device)  
    
    log_weights = torch.zeros(num_samples, device=device)
    cumulative_grad = torch.zeros(num_samples, device=device)
    k = 0.5


    for t in tqdm(torch.linspace(0, 1, num_time_steps)[1:]):
        
        time_gap = 1/(num_time_steps - 1)
        current_beta_k = t**k
        previous_beta_k = (t-time_gap)**k
        
        annealed_energy = AnnealedEnergy(annealed_densities, current_beta_k)
        sample = one_step_langevin_dynamic(
            sample, annealed_energy.log_reward, cfg.ld_step, do_correct=True
        )
        
        
        # Discrete Jarzynski estimator (log)
        log_current_energy_value = (1-current_beta_k)*(prior.log_reward(sample)) + current_beta_k*(target.log_reward(sample))
        log_previous_energy_value = (1-previous_beta_k)*(prior.log_reward(sample)) + previous_beta_k*(target.log_reward(sample))
        
        log_weight = log_current_energy_value - log_previous_energy_value 
        log_weights = log_weights + log_weight 
        
        
        # Continuous Jarzynski estimator
        t_tensor = torch.tensor([t], requires_grad=True, device=device).expand(sample.size(0)) 
        log_reward_t = (1-t_tensor)*(prior.log_reward(sample)) + t_tensor*(target.log_reward(sample))
        partial_log_p_t = torch.autograd.grad(log_reward_t.sum(), t_tensor)[0]
        cumulative_grad += partial_log_p_t*(current_beta_k-previous_beta_k) 


    # Jarzynski estimation
    discrete_jarzynski = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(num_samples))
    continuous_jarzynski = torch.logsumexp(cumulative_grad, dim=0) - torch.log(torch.tensor(num_samples))
    
    real_jarzynski_estimation = -torch.logsumexp(-log_weights, dim=0) + torch.log(torch.tensor(num_samples))
    
    # Log variance of exp(-Work)
    # discrete_max_log_weight = torch.max(log_weights)
    # discrete_log_var_Jarzynski = -2*discrete_max_log_weight + torch.log(torch.var(torch.exp(discrete_max_log_weight-log_weights)))
    
    # continuous_max_grad = torch.max(cumulative_grad)
    # continuous_log_var_Jarzynski = -2*continuous_max_grad + torch.log(torch.var(torch.exp(continuous_max_grad-cumulative_grad)))

    d_log_E_exp_neg_log_weights = torch.logsumexp(-log_weights, dim=0) - torch.log(torch.tensor(num_samples))
    d_log_E_exp_neg_2log_weights = torch.logsumexp(-2 * log_weights, dim=0) - torch.log(torch.tensor(num_samples))
    d_log_variance = d_log_E_exp_neg_2log_weights + torch.log(1 - torch.exp(2 * d_log_E_exp_neg_log_weights - d_log_E_exp_neg_2log_weights))
    
    c_log_E_exp_neg_work = torch.logsumexp(-cumulative_grad, dim=0) - torch.log(torch.tensor(num_samples))
    c_log_E_exp_neg_2work = torch.logsumexp(-2 * cumulative_grad, dim=0) - torch.log(torch.tensor(num_samples))
    c_log_variance = c_log_E_exp_neg_2work + torch.log(1 - torch.exp(2 * c_log_E_exp_neg_work - c_log_E_exp_neg_2work))


    print("log variance of exp(-Work) by AIS: ", d_log_variance.item())
    print("log variance of exp(-Work) by Integral: ", c_log_variance.item())
    print("Mean of exp(W) : ", torch.mean(log_weights).item())   
    print("Real Jarzynski estimation: ", real_jarzynski_estimation.item())

    return sample, discrete_jarzynski, continuous_jarzynski


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num_sample", type=int, required=True)
    parser.add_argument("-A", "--annealing_step", type=int, required=True)
    parser.add_argument("-T", "--MCMC_step", type=int, required=True)
    args = parser.parse_args()

    cfg = DictConfig(
        {
            "num_samples": args.num_sample,
            "num_time_steps": args.annealing_step,
            "max_iter_ls": args.MCMC_step,
            "burn_in": args.MCMC_step - 100,
            "ld_schedule": True,
            "ld_step": 0.01,
            "target_acceptance_rate": 0.574,
            "device": "cuda",
            "energy": {
                "_target_": "energy.gmm.GMM25",
                "dim": 2,
            },
            "eval": {
                "plot": {
                    "plotting_bounds": [-20.0, 20.0],
                    #"projection_dims": [[0, 2], [1, 2], [2, 4], [3, 4], [4, 6], [5, 6]],
                    "fig_size": [12, 12],
                }
            },
        }
    )

    energy = get_energy_function(cfg)
    prior = GaussianEnergy(device="cuda", dim=2, std=7.0)
    plotter = SamplePlotter(energy, **cfg.eval.plot)

    sample, discrete_jarzynski, continuous_jarzynski = annealed_IS_with_langevin(prior, energy, cfg)
    
    output_dir_sample = "results/figure/AIS-Langevin/sample"
    output_dir_energy = "results/figure/AIS-Langevin/energy"
    output_dir_weights = "results/figure/AIS-Langevin/weights"
    
    os.makedirs(output_dir_sample, exist_ok=True)
    os.makedirs(output_dir_energy, exist_ok=True)
    os.makedirs(output_dir_weights, exist_ok=True)
    
    config_postfix = f"N={args.num_sample}-A={args.annealing_step}-T={args.MCMC_step}"

    fig, ax = plotter.make_sample_plot(sample)
    fig.savefig(
        f"results/figure/AIS-Langevin/sample-{config_postfix}.pdf",
        bbox_inches="tight",
    )

    # fig, ax = plotter.make_energy_histogram(sample)
    # fig.savefig(
    #     f"results/figure/AIS-Langevin/energy-{config_postfix}.pdf",
    #     bbox_inches="tight",
    # )

    # Print mean of importance weights ~ log(Z_T/Z_0)
    print(f"Free energy estimation by AIS: {discrete_jarzynski.item()}") # Discrete Jarzynski estimator
    print(f"Free energy estimation by Integrated Work: {continuous_jarzynski.item()}") # Continuous Jarzynski estimator

    # Print difference of ground truth logZ b/w target & prior
    print(f"Prior ground truth logZ: {prior.ground_truth_logZ}")
    print(f"Target ground truth logZ: {energy.ground_truth_logZ}")
    print(f"(Ground truth) Free energy difference: {energy.ground_truth_logZ - prior.ground_truth_logZ}")
