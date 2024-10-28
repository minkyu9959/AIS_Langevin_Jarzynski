import cProfile

import torch
import hydra

from omegaconf import DictConfig

from train import set_seed
from models import get_model, SamplerModel, OldGFN
from energy import BaseEnergy, get_energy_function

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@hydra.main(
    version_base="1.3", config_path="../../../configs", config_name="profile.yaml"
)
def profile(cfg: DictConfig) -> None:
    if cfg.name is None:
        raise Exception("You should specify the model")

    set_seed(cfg.seed)

    energy_function: BaseEnergy = get_energy_function(cfg.energy, device=cfg.device)

    if cfg.name == "Sampler":
        model: SamplerModel = get_model(cfg, energy_function).to(cfg.device)
    elif cfg.name == "OldGFN":
        model = OldGFN(
            dim=energy_function.data_ndim,
            s_emb_dim=64,
            hidden_dim=64,
            harmonics_dim=64,
            t_dim=64,
            energy_function=energy_function,
            langevin=False,
            pis_architectures=False,
            learn_pb=False,
            zero_init=False,
            clipping=True,
            device=DEVICE,
        ).to(DEVICE)
    else:
        raise Exception("Invalid model name")

    def one_training_step():
        initial_states = model.generate_initial_state(300)
        _, log_pfs, log_pbs = model.get_forward_trajectory(initial_states)

        # Caution: this loss is not pratical.
        loss = 0.5 * ((log_pfs.sum(-1) - log_pbs.sum(-1)) ** 2).mean()

        loss.backward()

    cProfile.runctx("one_training_step()", globals(), locals(), sort="tottime")


if __name__ == "__main__":
    profile()
