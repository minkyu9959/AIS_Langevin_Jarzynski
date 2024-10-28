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
from energy.neural_drift import NeuralDrift
from utility import SamplePlotter

import wandb
from datetime import datetime


#@torch.no_grad()
def annealed_IS_with_langevin(prior: BaseEnergy, target: BaseEnergy, cfg: DictConfig):
    
    device = cfg.device
    annealing_steps = cfg.num_time_steps
    num_samples = cfg.num_samples
    num_epochs = 20000 #Minkyu
    max_norm = 30.0 #Minkyu
    
    input_dim = cfg.energy["dim"] #Minkyu
    sigma = 2.0 #Minkyu
    neural_drift = NeuralDrift(input_dim).to(device) #Minkyu
    
    TB_param = torch.nn.Parameter(torch.tensor(100.0, device=device)) #Minkyu
    
    annealed_densities = AnnealedDensities(target, prior) # (1-t)*E_0(x) + t*E_T(x)
    
    optimizer = torch.optim.Adam(
            [
                {"params": TB_param, "lr": 1e-2},
                {"params": neural_drift.parameters(), "lr": 1e-3},
            ]
        ) #Minkyu
    
    # Wandb setting
    wandb_config = {
        "device": str(cfg.device),
        "num_time_steps": cfg.num_time_steps,
        "num_samples": cfg.num_samples,
        "energy_dim": cfg.energy["dim"],
        "ld_step": cfg.ld_step
    } #Minkyu
    wandb.init(project="annealed_IS_with_langevin", config=wandb_config) #Minkyu
    
    F_gap = target.ground_truth_logZ - prior.ground_truth_logZ #Minkyu

    for step in tqdm(range(num_epochs)): #Minkyu
        
        # sample = prior.sample(num_samples, device) # prior sample = x_1
        sample = prior.sample(num_samples, device)
        
        # Annealed importance weight
        radon_nikodym = torch.zeros(num_samples, device=device) #Minkyu
        log_p_X = prior.log_reward(sample) - prior.ground_truth_logZ #Fix_backwards #P(X_0)

        for t in torch.linspace(0, 1, annealing_steps)[1:]:
            
            dt = 1/(annealing_steps - 1) #Minkyu
            
            annealed_energy = AnnealedEnergy(annealed_densities, t)
            
            # Sample update : x_t => x_{t+1}
            sample, log_transition_prob = one_step_langevin_dynamic(
                neural_drift, sample, t, annealed_energy.log_reward, cfg.ld_step, sigma, do_correct=True
            )
        
            infinitesimal_work = ( #Minkyu
                prior.energy(sample) - target.energy(sample) + (neural_drift(sample, t)**2).sum(dim=-1) 
            )
            
            radon_nikodym = radon_nikodym + infinitesimal_work * dt #Jarzynski work integration
            log_p_X = log_p_X + log_transition_prob #Fix_backwards #P(X_0) * P(X_1|X_0) * ... * P(X_T|X_{T-1})
            
        # Jarzynski estimation
        # unbiased_est = torch.logsumexp(work, dim=0) - torch.log(torch.tensor(num_samples)) #Minkyu
        
        # Train 
        E_radon_nikodym = radon_nikodym.mean()
        radon_nikodym_detached = radon_nikodym.detach().clone()
        E_reinforce_term = (radon_nikodym_detached * log_p_X).mean()
        
        # Compute the overall loss
        loss = -(E_radon_nikodym + E_reinforce_term) #-ELBO with reinforce trick
        # loss = (radon_nikodym - TB_param).square().mean() #Average learning TB loss with scalar parameter
        # loss = (radon_nikodym - 135).square().mean() #TB with constant
        # loss = (radon_nikodym - radon_nikodym.mean()).square().mean() #VarGrad MC estimation
        loss.backward() #Minkyu
        
        torch.nn.utils.clip_grad_norm_(neural_drift.parameters(), max_norm) #Minkyu
        
        total_norm = 0 #Minkyu
        for p in neural_drift.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5 #Minkyu
        
        optimizer.step() #Minkyu
        optimizer.zero_grad() #Minkyu
         
        # Wandb logging
        wandb.log({
            # "Unbiased estimation": (unbiased_est + prior.ground_truth_logZ).item(),
            "ELBO": (radon_nikodym.mean() + prior.ground_truth_logZ).item(),
            "TB_param": TB_param.item(), 
            "Loss": loss.item(),
            "Gradient Norm": total_norm,
        }) #Minkyu
        
        if step % 100 == 0: #Minkyu
        
            output_dir_sample = "results/figure/AIS-Langevin/sample"
            output_dir_energy = "results/figure/AIS-Langevin/energy"
            output_dir_weights = "results/figure/AIS-Langevin/weights"
        
            os.makedirs(output_dir_sample, exist_ok=True)
            os.makedirs(output_dir_energy, exist_ok=True)
            os.makedirs(output_dir_weights, exist_ok=True)
        
            config_postfix = f"({step} per {num_epochs})"

            fig, ax = plotter.make_sample_plot(sample)
            fig.savefig(
                f"results/figure/AIS-Langevin/sample-{config_postfix}.pdf",
                bbox_inches="tight",
            ) #Minkyu

    return sample, unbiased_est


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num_sample", type=int, required=True)
    parser.add_argument("-A", "--annealing_step", type=int, required=True)
    args = parser.parse_args()

    cfg = DictConfig(
        {
            "num_samples": args.num_sample,
            "num_time_steps": args.annealing_step,
            "ld_schedule": True,
            "ld_step": 0.01,
            "target_acceptance_rate": 0.574,
            "device": "cuda",
            "energy": {
                "_target_": "energy.many_well.ManyWell",
                "dim": 32,
            },
            "eval": {
                "plot": {
                    "plotting_bounds": [-3.0, 3.0],
                    "projection_dims": [[0, 2], [1, 2], [2, 4], [3, 4], [4, 6], [5, 6]],
                    "fig_size": [12, 20],
                }
            },
        }
    )

    energy = get_energy_function(cfg)
    prior = GaussianEnergy(device="cuda", dim=32, std=1.0)
    plotter = SamplePlotter(energy, **cfg.eval.plot)

    sample, unbiased_est = annealed_IS_with_langevin(prior, energy, cfg)
    
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

    fig, ax = plotter.make_energy_histogram(sample)
    fig.savefig(
        f"results/figure/AIS-Langevin/energy-{config_postfix}.pdf",
        bbox_inches="tight",
    )

    # Print mean of importance weights ~ log(Z_T/Z_0)
    print(f"Free energy estimation by AIS: {unbiased_est.item()}") # Discrete unbiased estimation
    
    # Print difference of ground truth logZ b/w target & prior
    print(f"Prior ground truth logZ: {prior.ground_truth_logZ}")
    print(f"Target ground truth logZ: {energy.ground_truth_logZ}")
    print(f"(Ground truth) Free energy difference: {energy.ground_truth_logZ - prior.ground_truth_logZ}")