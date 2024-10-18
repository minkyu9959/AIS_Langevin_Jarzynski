from .base_model import SamplerModel

from .old_gfn_v1 import GFN as OldGFNv1
from .old_gfn import GFN as OldGFN

from .cmcd import CMCDSampler
from .gfn import GFN
from .annealed_gfn import AnnealedGFN
from .double_gfn import DoubleGFN

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig

from energy import BaseEnergy
from energy import NeuralEnergy #Chaehyeon

def get_model(cfg: DictConfig, energy_function: BaseEnergy, neural_energy: NeuralEnergy) -> SamplerModel:
    model = instantiate(
        cfg.model,
        device=torch.device(cfg.device),
        energy_function=energy_function,
        neural_energy=neural_energy, #Chaehyeon
    )

    return model


__all__ = [
    "get_model",
    "SamplerModel",
    "GFN",
    "AnnealedGFN",
    "CMCDSampler",
    "DoubleGFN",
]
