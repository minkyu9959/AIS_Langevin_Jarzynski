"""
Train code for GFN with local search buffer + Langevin parametrization
(Sendera et al., 2024, Improved off-policy training of diffusion samplers)
"""

from buffer import *

from trainer import BaseTrainer

from models.loss import get_forward_loss, get_backward_loss

from .utils.gfn_utils import (
    get_buffer,
    get_exploration_std,
)
from .utils.langevin import langevin_dynamics


def get_exploration_schedule(train_cfg, epoch: int):
    return get_exploration_std(
        epoch=epoch,
        exploratory=train_cfg.exploratory,
        exploration_factor=train_cfg.exploration_factor,
        exploration_wd=train_cfg.exploration_wd,
    )


class OnPolicyTrainer(BaseTrainer):
    def initialize(self):
        self.loss_fn = get_forward_loss(self.train_cfg.fwd_loss)

    def train_step(self) -> float:
        self.model.zero_grad()

        loss = self.loss_fn(
            gfn=self.model,
            batch_size=self.train_cfg.batch_size,
            exploration_schedule=get_exploration_schedule(
                self.train_cfg, self.current_epoch
            ),
        )

        loss.backward()
        self.optimizer.step()
        return loss.item()


class OffPolicyTrainer(BaseTrainer):
    def initialize(self):
        self.set_buffer()

        self.fwd_loss_fn = get_forward_loss(self.train_cfg.fwd_loss)
        self.bwd_loss_fn = get_backward_loss(self.train_cfg.bwd_loss)

    def set_buffer(self):
        train_cfg = self.train_cfg
        self.buffer = get_buffer(train_cfg.buffer, self.energy_function)
        self.local_search_buffer = get_buffer(train_cfg.buffer, self.energy_function)

    def train_step(self) -> float:
        self.model.zero_grad()
        exploration_std = get_exploration_schedule(self.train_cfg, self.current_epoch)

        train_cfg = self.train_cfg

        # For even epoch, train with forward trajectory
        if self.current_epoch % 2 == 0:
            loss, states, _, _, log_r = self.fwd_loss_fn(
                self.model,
                batch_size=train_cfg.batch_size,
                exploration_schedule=exploration_std,
                return_experience=True,
            )

            self.buffer.add(states[:, -1], log_r)

        # For odd epoch, train with backward trajectory
        else:
            samples = self.sample_from_buffer()
            loss = self.bwd_loss_fn(
                self.model,
                samples,
            )

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sample_from_buffer(self):
        train_cfg = self.train_cfg
        epoch = self.current_epoch
        energy = self.energy_function

        if train_cfg.get("local_search"):
            if epoch % train_cfg.local_search.ls_cycle < 2:
                samples, rewards = self.buffer.sample()
                local_search_samples, log_r = langevin_dynamics(
                    samples,
                    energy.log_reward,
                    train_cfg.device,
                    train_cfg.local_search,
                )
                self.local_search_buffer.add(local_search_samples, log_r)

            samples, rewards = self.local_search_buffer.sample()
        else:
            samples, rewards = self.buffer.sample()

        return samples


class SampleBasedTrainer(BaseTrainer):
    def initialize(self):
        self.loss_fn = get_backward_loss(self.train_cfg.bwd_loss)

    def train_step(self) -> float:
        self.model.zero_grad()

        train_cfg = self.train_cfg
        energy = self.energy_function

        samples = energy.sample(train_cfg.batch_size, device=train_cfg.device)

        loss = self.loss_fn(self.model, sample=samples)

        loss.backward()
        self.optimizer.step()

        return loss.item()