from .base_energy import BaseEnergy, HighDimensionalEnergy
from .funnel import Funnel
from .many_well import ManyWell
from .gmm import GaussianMixture, GMM9, GMM25, GMM40

from .annealed_energy import AnnealedEnergy, AnnealedDensities
from .simple_energy import GaussianEnergy, UniformEnergy, DiracDeltaEnergy
from models.neural_drift import NeuralDrift

from hydra.utils import instantiate
from omegaconf import DictConfig


def get_energy_function(cfg: DictConfig) -> BaseEnergy:
    energy_function = instantiate(cfg.energy, device=cfg.device)

    return energy_function
