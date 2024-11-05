import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from omegaconf import DictConfig

from tqdm import tqdm
from argparse import ArgumentParser

from trainer.utils.langevin import one_step_langevin_dynamic, langevin_proposal_bwd, get_reward_and_gradient
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

from models.loss import compute_loss


#@torch.no_grad()
def annealed_IS_with_langevin(prior: BaseEnergy, target: BaseEnergy, cfg: DictConfig):
    
    device = cfg.device
    annealing_steps = cfg.num_time_steps
    num_samples = cfg.num_samples
    num_epochs = 10000 
    max_norm = 10.0 
    
    input_dim = cfg.energy["dim"] 
    sigma = 2.0 
    neural_drift = NeuralDrift(input_dim).to(device) 
    
    TB_param = torch.nn.Parameter(torch.tensor(100.0, device=device))
    
    annealed_densities = AnnealedDensities(target, prior) # (1-t)*E_0(x) + t*E_T(x)
    
    optimizer = torch.optim.Adam(
            [
                {"params": TB_param, "lr": 1e-1},
                {"params": neural_drift.parameters(), "lr": 1e-3},
            ]
        ) 
    
    # Wandb setting
    # wandb_config = {
    #     "device": str(cfg.device),
    #     "num_time_steps": cfg.num_time_steps,
    #     "num_samples": cfg.num_samples,
    #     "energy_dim": cfg.energy["dim"],
    #     "ld_step": cfg.ld_step
    # } 
    # wandb.init(project="annealed_IS_with_langevin", config=wandb_config) 

    for step in tqdm(range(num_epochs)): 
        
        # sample = prior.sample(num_samples, device) # prior sample = x_1
        sample = prior.sample(num_samples, device)
        
        # Annealed importance weight
        radon_nikodym = torch.zeros(num_samples, device=device)
        log_p_X = prior.log_reward(sample) - prior.ground_truth_logZ #Prior(X_0)
        log_weights = prior.log_reward(sample) #Reverse KL in VAE

        for t in torch.linspace(0, 1, annealing_steps)[1:]:
            
            dt = 1/(annealing_steps - 1) 
            
            annealed_energy = AnnealedEnergy(annealed_densities, t)
            
            infinitesimal_work = ( 
                - prior.energy(sample) + target.energy(sample) + (neural_drift(sample, t)**2).sum(dim=-1) 
            )
            
            # Sample update : x_t => x_{t+1}
            sample, log_p_transition_fwd, log_p_transition_bwd, log_importance_weight = one_step_langevin_dynamic(
                neural_drift, sample, t, annealed_energy.log_reward, cfg.ld_step, sigma, do_correct=True
            )
            
            radon_nikodym = radon_nikodym + infinitesimal_work * dt #Jarzynski work integration
            log_p_X = log_p_X + log_p_transition_fwd #P(X_0) * P(X_1|X_0) * ... * P(X_T|X_{T-1})
            log_weights = log_weights + log_importance_weight #Reverse KL in VAE
        
        radon_nikodym = radon_nikodym + (- prior.energy(sample) + target.energy(sample) + (neural_drift(sample, t)**2).sum(dim=-1)) * dt 
        
        log_weights = log_weights - target.log_reward(sample)
        reverse_KL_VAE = log_weights.mean() + target.ground_truth_logZ #Reverse KL in VAE
        
        lower_bound_VAE = -reverse_KL_VAE + target.ground_truth_logZ #ELBO with reparametrization trick (VAE)
        
        reverse_KL_Jarzynski = radon_nikodym.mean() + target.ground_truth_logZ - prior.ground_truth_logZ #Reverse KL in Jarzynski
        lower_bound_Jarzynski = -reverse_KL_Jarzynski + target.ground_truth_logZ #ELBO with reparametrization trick (Jarzynski)
        
        print("Lower Bound (VAE): ", lower_bound_VAE.item())
        print("Lower Bound (Jarzynski): ", lower_bound_Jarzynski.item())
        
        # Train 
        
        loss = compute_loss(cfg.loss_type, log_weights=log_weights, radon_nikodym=radon_nikodym, target_log_Z=target.ground_truth_logZ, prior_log_Z=prior.ground_truth_logZ, TB_param=TB_param)
        
        loss.backward() 
        
        torch.nn.utils.clip_grad_norm_(neural_drift.parameters(), max_norm) 
        
        total_norm = 0 
        for p in neural_drift.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5 
        
        optimizer.step() 
        optimizer.zero_grad() 
        
        # Wandb logging
        # wandb.log({
        #     "Lower Bound (VAE)": lower_bound_VAE.item(),
        #     "Lower Bound (Jarzynski)": lower_bound_Jarzynski.item(),
        #     "TB_param": TB_param.item(), 
        #     "Loss": loss.item(),
        #     "Gradient Norm": total_norm,
        # }) 
        
        if step % 100 == 0:
        
            output_dir_sample = "results/figure/AIS-Langevin/sample"
            output_dir_energy = "results/figure/AIS-Langevin/energy"
            output_dir_weights = "results/figure/AIS-Langevin/weights"
        
            os.makedirs(output_dir_sample, exist_ok=True)
            os.makedirs(output_dir_energy, exist_ok=True)
            os.makedirs(output_dir_weights, exist_ok=True)
        
            config_postfix = f"TB_avg ({step} per {num_epochs})"

            fig, ax = plotter.make_sample_plot(sample)
            fig.savefig(
                f"results/figure/AIS-Langevin/sample-{config_postfix}.pdf",
                bbox_inches="tight",
            ) 

    return sample, unbiased_est


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num_sample", type=int, required=True)
    parser.add_argument("-A", "--annealing_step", type=int, required=True)
    parser.add_argument("-L", "--loss_type", type=str, choices=["elbo_jarzynski", "elbo_vae", "vargrad", "tb_avg"], required=True)
    
    args = parser.parse_args()

    cfg = DictConfig(
        {
            "num_samples": args.num_sample,
            "num_time_steps": args.annealing_step,
            "loss_type": args.loss_type,
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

    # Print mean of importance weights ~ log(Z_T/Z_0)
    print(f"Free energy estimation by AIS: {unbiased_est.item()}") # Discrete unbiased estimation
    
    # Print difference of ground truth logZ b/w target & prior
    print(f"Prior ground truth logZ: {prior.ground_truth_logZ}")
    print(f"Target ground truth logZ: {energy.ground_truth_logZ}")
    print(f"(Ground truth) Free energy difference: {energy.ground_truth_logZ - prior.ground_truth_logZ}")