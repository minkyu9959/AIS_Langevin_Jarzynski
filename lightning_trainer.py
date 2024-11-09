import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig
import yaml
from argparse import ArgumentParser

import lightning as L
import torch

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

from models.loss import compute_loss
from models.metric import compute_lower_bound

class AnnealedLangevinLightning(L.LightningModule):
    def __init__(self, prior: BaseEnergy, target: BaseEnergy, cfg: DictConfig):
        super().__init__()
        self.prior = prior
        self.target = target
        self.loss_type = cfg.loss_type
        
        self.device = cfg.device
        self.input_dim = cfg.energy["dim"]
        self.annealing_steps = cfg.num_time_steps
        self.step_size = 1/self.annealing_steps
        self.num_samples = cfg.num_samples
        
        self.num_epochs = 10000
        self.max_norm = cfg.max_norm
        self.sigma = cfg.sigma
        
        
        self.neural_drift = NeuralDrift(self.input_dim).to(self.device)
        self.TB_param = torch.nn.Parameter(torch.tensor(100.0, device=self.device))
        
        self.annealed_densities = AnnealedDensities(self.target, self.prior)

    def forward(self, x, t):
        return self.neural_drift(x, t)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.TB_param, "lr": 1e-1},
                {"params": self.neural_drift.parameters(), "lr": 1e-3},
            ]
        )
        return optimizer
    
    def training_step(self, radon_nikodym, log_weights):
        # Prior samples X_0
        sample = self.prior.sample(self.num_samples, self.device)
        
        radon_nikodym = torch.zeros(self.num_samples, device=self.device)
        log_weights = -self.prior.energy(sample)
        
        log_p_X = -self.prior.energy(sample) - self.prior.ground_truth_logZ
        
        for t in torch.linspace(0, 1, self.annealing_steps)[1:]:
            dt = self.step_size
            
            annealed_energy = AnnealedEnergy(self.annealed_densities, t)
            
            infinitesimal_work = (
                - self.prior.energy(sample) + self.target.energy(sample) + (self.neural_drift(sample, t) ** 2).sum(dim=-1)
            )
            # Jarzynski work integral
            radon_nikodym = radon_nikodym + infinitesimal_work * dt
            
            # Langevin dynamics for next states (t+1 step's samples)
            sample, log_p_transition_fwd, log_importance_weight = one_step_langevin_dynamic(
                self.neural_drift, sample, t, annealed_energy.log_reward, dt, self.sigma
            )
            # Importance sampling weights (forward transition prob / backward transition prob)
            log_weights = log_weights + log_importance_weight
            
            # Path measure log P(X) for reinforce trick
            log_p_X = log_p_X + log_p_transition_fwd
        
        radon_nikodym = radon_nikodym + (- self.prior.energy(sample) + self.target.energy(sample) + (self.neural_drift(sample, t) ** 2).sum(dim=-1)) * dt
        log_weights = log_weights - self.target.log_reward(sample)
        
        loss = compute_loss(
            self.loss_type, log_weights=log_weights, radon_nikodym=radon_nikodym,
            target_log_Z=self.target.ground_truth_logZ, prior_log_Z=self.prior.ground_truth_logZ, TB_param=self.TB_param, log_p_X=log_p_X
        )
        lower_bound = compute_lower_bound(self.loss_type, log_weights=log_weights, prior_log_Z=self.prior.ground_truth_logZ)
        
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_ELBO", lower_bound, on_epoch=True)

        return loss

def train_model(prior, target, cfg):
    model = AnnealedLangevinLightning(prior, target, cfg)
    trainer = L.Trainer(max_epochs=10000, gpus=1 if torch.cuda.is_available() else 0, gradient_clip_val=cfg.max_norm) 
    trainer.fit(model)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num_sample", type=int, required=True)
    parser.add_argument("-A", "--annealing_step", type=int, required=True)
    parser.add_argument("-L", "--loss_type", type=str, choices=["reverse_KL_vae", "reverse_KL_jarzynski", "vargrad", "tb_avg"], required=True)
    parser.add_argument("-T", "--target", type=str, required=True, help="Target energy configuration file, e.g., ManyWell or GMM25")

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
        "max_norm": 1.0,
        "sigma": 2.0,
    }

    # Update the configuration using OmegaConf
    cfg = OmegaConf.merge(DictConfig(base_cfg), DictConfig(cmd_cfg))

    # Rest of the code as in the original script
    target = get_energy_function(cfg)
    prior = GaussianEnergy(device="cuda", dim=cfg.energy.dim, std=cfg.model.prior_energy.std)
    plotter = SamplePlotter(target, **cfg.eval.plot)
    
    train_model(prior, target, cfg)