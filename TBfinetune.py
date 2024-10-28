import torch

from utility.loader import load_all_from_experiment_path


PATH = "/home/guest_dyw/diffusion-sampler/results/outputs/2024-09-08/11-35-56"


def main() -> None:
    """
    Main entry point for finetuning.
    """
    energy, model, trainer, _ = load_all_from_experiment_path(
        PATH, must_init_logger=True
    )

    trainer.train_cfg.fwd_loss = "tb"
    trainer.train_cfg.bwd_loss = "tb"
    trainer.max_epoch = trainer.train_cfg.epochs = 3000

    model.logZ = torch.nn.Parameter(
        model.logZ_ratio.detach().sum(dim=0) + model.prior_energy.ground_truth_logZ
    )

    trainer.train()


if __name__ == "__main__":
    main()
