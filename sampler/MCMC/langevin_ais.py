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
from energy.annealed_energy import NeuralEnergy

import wandb
from datetime import datetime


#@torch.no_grad()
def annealed_IS_with_langevin(prior: BaseEnergy, target: BaseEnergy, cfg: DictConfig):
    
    device = cfg.device
    num_time_steps = cfg.num_time_steps
    num_samples = cfg.num_samples
    num_epochs = 10000 #Minkyu
    lr = 1e-5 #Minkyu
    max_norm = 2.0 #Minkyu
    
    input_dim = cfg.energy["dim"] #Minkyu
    neural_energy = NeuralEnergy(input_dim).to(device) #Minkyu
    ELBO_est_param = torch.nn.Parameter(torch.tensor(100.0, device=device)) #Minkyu
    
    annealed_densities = AnnealedDensities(target, prior, neural_energy) # (1-t)*E_0(x) + t*E_T(x) + t*(1-t)*NN(x)
    
    optimizer = torch.optim.Adam(
            [
                {"params": ELBO_est_param, "lr": 1e-2},
                {"params": neural_energy.parameters(), "lr": lr, "weight_decay": 1e-4},
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
        log_weights = torch.zeros(num_samples, device=device) #Minkyu

        for t in torch.linspace(0, 1, num_time_steps)[1:]:
            
            dt = 1/(num_time_steps - 1) #Minkyu
            prev_t = t - dt #Minkyu
            
            annealed_energy = AnnealedEnergy(annealed_densities, t) 
            prev_annealed_energy = AnnealedEnergy(annealed_densities, prev_t) #Minkyu
            
            # Sample update : x_t => x_{t+1}
            sample = one_step_langevin_dynamic(
                sample, annealed_energy.log_reward, cfg.ld_step, do_correct=True
            )
            
            log_weights = log_weights + (annealed_energy.log_reward(sample) - prev_annealed_energy.log_reward(sample)) #Minkyu

        # Jarzynski estimation
        unbiased_est = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(num_samples)) #Minkyu
        
        # Train
        # ELBO_est = (log_weights).mean()
        loss = (ELBO_est_param - log_weights).square().mean() # Var[log_weights] #Minkyu
        
        # l2_regularization = 0.0
        # for param in neural_energy.parameters():
        #     l2_regularization += torch.norm(param, 2)  # L2 reg

        # # Add the L2 regularization to the loss (scaled by a regularization factor)
        # reg_weight = 0.1  # Adjust as necessary
        # reg_loss = loss + reg_weight * l2_regularization
        
        loss.backward() #Minkyu
        
        torch.nn.utils.clip_grad_norm_(neural_energy.parameters(), max_norm) #Minkyu
         
        total_norm = 0 #Minkyu
        for p in neural_energy.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5 #Minkyu
        
        optimizer.step() #Minkyu
        optimizer.zero_grad() #Minkyu
         
        # Wandb logging
        wandb.log({
            "Unbiased estimation": unbiased_est.item(),
            "ELBO": log_weights.mean().item(),
            "ELBO_estimation": ELBO_est_param.item(), 
            "Loss": loss.item(),
            # "L2 Regularization": l2_regularization.item(),
            "Gradient Norm": total_norm,
        }) #Minkyu
        
        if step % 100 == 0: #Minkyu
        
            output_dir_sample = "results/figure/AIS-Langevin/sample"
            output_dir_energy = "results/figure/AIS-Langevin/energy"
            output_dir_weights = "results/figure/AIS-Langevin/weights"
        
            os.makedirs(output_dir_sample, exist_ok=True)
            os.makedirs(output_dir_energy, exist_ok=True)
            os.makedirs(output_dir_weights, exist_ok=True)
        
            config_postfix = f"A={num_time_steps}, max_norm={max_norm}, lr={lr} ({step} per {num_epochs})"

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

    # fig, ax = plotter.make_energy_histogram(sample)
    # fig.savefig(
    #     f"results/figure/AIS-Langevin/energy-{config_postfix}.pdf",
    #     bbox_inches="tight",
    # )

    # Print mean of importance weights ~ log(Z_T/Z_0)
    print(f"Free energy estimation by AIS: {unbiased_est.item()}") # Discrete unbiased estimation
    
    # Print difference of ground truth logZ b/w target & prior
    print(f"Prior ground truth logZ: {prior.ground_truth_logZ}")
    print(f"Target ground truth logZ: {energy.ground_truth_logZ}")
    print(f"(Ground truth) Free energy difference: {energy.ground_truth_logZ - prior.ground_truth_logZ}")