import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from omegaconf import OmegaConf, DictConfig
import yaml

from tqdm import tqdm
from argparse import ArgumentParser

from langevin import one_step_langevin_dynamic
from energy import (
    AnnealedDensities,
    AnnealedEnergy,
    BaseEnergy,
    get_energy_function,
    GaussianEnergy,
)
from models.neural_drift import NeuralDrift
from utility import SamplePlotter

import wandb

from models.loss import compute_loss
from models.metric import compute_lower_bound


#@torch.no_grad()
def annealed_IS_with_langevin(prior: BaseEnergy, target: BaseEnergy, cfg: DictConfig):
    
    device = cfg.device
    annealing_steps = cfg.num_time_steps
    step_size = 1/(annealing_steps-1)
    num_samples = cfg.num_samples
    num_epochs = 50000 
    max_norm = cfg.max_norm
    reinforce = cfg.reinforce
    
    input_dim = cfg.energy["dim"] 
    sigma = cfg.sigma
    neural_drift = NeuralDrift(input_dim).to(device) 
    
    TB_param = torch.nn.Parameter(torch.tensor(0.0, device=device))
    
    annealed_densities = AnnealedDensities(target, prior) # (1-t)*E_0(x) + t*E_T(x)
    
    optimizer = torch.optim.Adam(
            [
                {"params": TB_param, "lr": 1e-1},
                {"params": neural_drift.parameters(), "lr": 1e-3},
            ]
        ) 
    
    # Wandb setting
    wandb_config = {
        "device": str(cfg.device),
        "num_time_steps": cfg.num_time_steps,
        "num_samples": cfg.num_samples,
        "energy_dim": cfg.energy["dim"],
    } 
    wandb.init(project="annealed_IS_with_langevin", config=wandb_config) 

    for epoch in tqdm(range(num_epochs)): 
        
        # sample = prior.sample(num_samples, device) # prior sample = x_1
        sample = prior.sample(num_samples, device)
        
        radon_nikodym = torch.zeros(num_samples, device=device) #Reverse KL in Jarzynski
        log_weights = -prior.energy(sample) #Reverse KL in VAE
        log_p_X = -prior.energy(sample) - prior.ground_truth_logZ #P(X_0)

        for t in torch.linspace(0, 1, annealing_steps)[1:]:
            
            dt = step_size
            
            annealed_energy = AnnealedEnergy(annealed_densities, t)
            
            infinitesimal_work = ( 
                - prior.energy(sample) + target.energy(sample) + (neural_drift(sample, t)**2).sum(dim=-1) 
            )
            
            radon_nikodym = radon_nikodym + infinitesimal_work * dt #Jarzynski work integration
            
            # Sample update : x_t => x_{t+1}
            sample, log_p_transition_fwd, log_importance_weight = one_step_langevin_dynamic(
                neural_drift, sample, t, annealed_energy.log_reward, dt, sigma, reinforce
            )
            
            log_weights = log_weights + log_importance_weight #Reverse KL in VAE
            log_p_X = log_p_X + log_p_transition_fwd #P(X_0) * P(X_1|X_0) * ... * P(X_T|X_{T-1})
        
        radon_nikodym = radon_nikodym + (- prior.energy(sample) + target.energy(sample) + (neural_drift(sample, t)**2).sum(dim=-1)) * dt 
        log_weights = log_weights + target.energy(sample)
        
        # Train 
        loss = compute_loss(cfg.loss_type, log_weights=log_weights, radon_nikodym=radon_nikodym, target_log_Z=target.ground_truth_logZ, prior_log_Z=prior.ground_truth_logZ, TB_param=TB_param, log_p_X=log_p_X)
        lower_bound = compute_lower_bound(cfg.loss_type, log_weights=log_weights, radon_nikodym=radon_nikodym, prior_log_Z=prior.ground_truth_logZ)
        loss.backward() 
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(neural_drift.parameters(), max_norm) 
        
        # Gradient norm computation
        total_norm = 0 
        for p in neural_drift.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5 
        
        # Parameter update
        optimizer.step() 
        optimizer.zero_grad() 
        
        # Wandb logging
        wandb.log({
            "Lower Bound": lower_bound.item(),
            "TB_param": TB_param.item(), 
            "Loss": loss.item(),
            "Gradient Norm": total_norm,
        }) 

    return sample


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num_sample", type=int, required=True)
    parser.add_argument("-A", "--annealing_step", type=int, required=True)
    parser.add_argument("-L", "--loss_type", type=str, choices=["KL_vae", "KL_jarzynski", "KL_jarzynski_reinforce", "vargrad", "tb_avg"], required=True)
    parser.add_argument("-T", "--target", type=str, required=True, help="ManyWell or GMM25")

    args = parser.parse_args()

    # Load the corresponding config file based on target energy
    config_path = f"configs/energy/{args.target}.yaml"
    with open(config_path, "r") as file:
        base_cfg = yaml.safe_load(file)

    # Merge the loaded config with the command-line arguments
    cmd_cfg = {
        "num_samples": args.num_sample,
        "num_time_steps": args.annealing_step,
        "loss_type": args.loss_type,
        "device": "cuda",
        "max_norm": 5.0,
        "sigma": 2.0,
        "reinforce": "reinforce" in args.loss_type
    }

    # Update the configuration using OmegaConf
    cfg = OmegaConf.merge(DictConfig(base_cfg), DictConfig(cmd_cfg))

    # Rest of the code as in the original script
    target = get_energy_function(cfg)
    prior = GaussianEnergy(device="cuda", dim=cfg.energy.dim, std=cfg.model.prior_energy.std)
    
    plotter = SamplePlotter(target, **cfg.eval.plot)

    sample = annealed_IS_with_langevin(prior, target, cfg)
    
    # Print difference of ground truth logZ b/w target & prior
    print(f"Prior ground truth logZ: {prior.ground_truth_logZ}")
    print(f"Target ground truth logZ: {target.ground_truth_logZ}")
    print(f"(Ground truth) Free energy difference: {target.ground_truth_logZ - prior.ground_truth_logZ}")