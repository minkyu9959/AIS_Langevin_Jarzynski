from functools import cache

import torch
import numpy as np

from .base_energy import BaseEnergy
    
class AnnealedDensities:
    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy, 
    ):
        self.energy_function = energy_function
        self.device = energy_function.device
        self.prior_energy = prior_energy

    def energy(self, times: torch.Tensor, states: torch.Tensor):

        prior_energy = self.prior_energy.energy(states)
        energy = self.energy_function.energy(states)

        return (1 - times) * prior_energy + times * energy

    def score(self, times: torch.Tensor, states: torch.Tensor):

        prior_score = self.prior_energy.score(states)
        target_score = self.energy_function.score(states)

        return (1 - times) * prior_score + times * target_score + times

    @cache
    @torch.no_grad()
    def logZ_t(self, num_samples: int, trajectory_length: int) -> torch.Tensor:
        """
        Estimate intermediate logZ with importance sampling.

        Args:
            intermediate_energy: AnnealedEnergy
            num_samples: int
                Number of samples to use for importance sampling.
            trajectory_length: int
                Number of step to log Z ratio estimation is performed.

        Returns:
            Tensor: Estimated intermediate logZ's.
        """
        target_log_Z = self.energy_function.ground_truth_logZ
        prior_log_Z = self.prior_energy.ground_truth_logZ

        sample = self._sample_from_importance_distribution(num_samples)
        sample_energy = self._importance_distribution_energy(sample)

        sample_dist_log_Z = target_log_Z + prior_log_Z + np.log(2)

        times = torch.linspace(0.0, 1.0, trajectory_length + 1, device=sample.device)[
            ..., None
        ]

        annealed_energy = self.energy(times, sample)

        log_partition = (
            torch.logsumexp(-annealed_energy + sample_energy, dim=1)
            - torch.log(torch.tensor(num_samples))
            + sample_dist_log_Z
        )

        return log_partition

    def logZ_ratios(self, num_samples: int, trajectory_length: int) -> torch.Tensor:
        """
        Estimate logZ ratio for each intermediate step.

        Args:
            num_samples (int): number of samples to use for importance sampling.
            trajectory_length (int): number of step to log Z ratio estimation is performed.

        Returns:
            Tensor: logZ ratio values.
        """

        logZ_t = self.logZ_t(num_samples, trajectory_length)

        return logZ_t[1:] - logZ_t

    def _sample_from_importance_distribution(self, num_samples: int) -> torch.Tensor:
        prior_sample = self.prior_energy.sample(num_samples)
        target_sample = self.energy_function.sample(num_samples)

        mask = torch.randint(
            0, 2, (num_samples,), device=prior_sample.device
        ).unsqueeze(-1)

        return prior_sample * mask + target_sample * (1 - mask)

    def _importance_distribution_energy(self, sample: torch.Tensor) -> torch.Tensor:
        sample_prior_energy = self.prior_energy.energy(sample)
        sample_target_energy = self.energy_function.energy(sample)

        target_log_Z = self.energy_function.ground_truth_logZ
        prior_log_Z = self.prior_energy.ground_truth_logZ

        return -torch.logsumexp(
            torch.stack(
                [
                    -sample_prior_energy + target_log_Z,
                    -sample_target_energy + prior_log_Z,
                ],
            ),
            dim=0,
        )


class AnnealedEnergy(BaseEnergy):
    logZ_is_available = False
    can_sample = False

    def __init__(self, density_family: AnnealedDensities, time: float):
        target_energy = density_family.energy_function

        super().__init__(target_energy.device, target_energy.ndim)

        self.annealed_targets = density_family
        self._time = time

    def energy(self, states: torch.Tensor):
        return self.annealed_targets.energy(self._time, states) 

    def score(self, states: torch.Tensor):
        return self.annealed_targets.score(self._time, states)

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        raise Exception("Cannot sample from annealed energy")
