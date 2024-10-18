import random

import torch
import numpy as np

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from energy import BaseEnergy, get_energy_function, NeuralEnergy #Chaehyeon
from models import get_model

from trainer import BaseTrainer
from utility.logger import get_logger, Logger

from configs.util import *


def train(
    cfg: DictConfig, model: torch.nn.Module, energy_function: BaseEnergy, logger: Logger
):
    trainer: BaseTrainer = instantiate(
        cfg.train.trainer,
        model=model,
        energy_function=energy_function,
        train_cfg=cfg.train,
        eval_cfg=cfg.eval,
        logger=logger,
    )

    trainer.train()


@hydra.main(version_base="1.3", config_path="./configs", config_name="main.yaml") 
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    check_config_and_set_read_only(cfg)

    set_seed(cfg.seed)

    energy_function: BaseEnergy = get_energy_function(cfg)

    #Chaehyeon (Neural Energy 인스턴스 생성)
    neural_energy: NeuralEnergy = NeuralEnergy(input_dim=cfg.model.neural_energy.input_dim)   

    model: torch.nn.Module = get_model(cfg, energy_function, neural_energy).to(cfg.device)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger: Logger = get_logger(cfg, output_dir, debug=cfg.get("detail_log", False))

    train(cfg, model, energy_function, logger)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


if __name__ == "__main__":
    main()
