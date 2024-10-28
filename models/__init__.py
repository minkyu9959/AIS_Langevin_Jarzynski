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

def get_model(cfg: DictConfig, energy_function: BaseEnergy) -> SamplerModel:
    model = instantiate(
        cfg.model,
        device=torch.device(cfg.device),
        energy_function=energy_function,
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
