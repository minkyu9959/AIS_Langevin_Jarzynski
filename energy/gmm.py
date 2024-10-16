import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_energy import BaseEnergy


class GaussianMixture(BaseEnergy):
    """
    Two dimensional Gaussian mixture distribution with same std deviation.
    """

    logZ_is_available = True
    _ground_truth_logZ = 0.0

    can_sample = True

    def __init__(
        self,
        device: str,
        dim: int,
        mode_list: torch.Tensor,
        scale: float = 1.0,
    ):
        assert dim == 2
        super().__init__(device=device, dim=dim)

        self._make_gmm_distribution(mode_list, scale)

    def _make_gmm_distribution(self, modes: torch.Tensor, scale: float):
        assert modes.ndim == 2 and modes.shape[1] == 2

        num_modes = len(modes)

        comp = D.Independent(
            D.Normal(modes, torch.ones_like(modes) * scale),
            1,
        )
        mix = D.Categorical(torch.ones(num_modes, device=self.device))
        self.gmm = MixtureSameFamily(mix, comp)

    @BaseEnergy._match_device
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.gmm.log_prob(x)

    @BaseEnergy._match_device
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.gmm.log_prob(x)

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return self.gmm.sample((batch_size,))


class GMM1(GaussianMixture):
    def __init__(self, device: str, dim: int):
        mode_list = torch.tensor([(0.0, 0.0)], device=device)

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=1.0,
        )


class GMM9(GaussianMixture):
    def __init__(self, device: str, dim: int, scale: float = 0.5477222):

        mode_list = torch.tensor(
            [(a, b) for a in [-5.0, 0.0, 5.0] for b in [-5.0, 0.0, 5.0]],
            device=device,
        )

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=scale,
        )


class GMM25(GaussianMixture):
    def __init__(self, device: str, dim: int, scale: float = 0.3):

        mode_list = torch.tensor(
            [
                (a, b)
                for a in [-10.0, -5.0, 0.0, 5.0, 10.0]
                for b in [-10.0, -5.0, 0.0, 5.0, 10.0]
            ],
            device=device,
        )

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=scale,
        )


class GMM40(GaussianMixture): #Minkyu
    def __init__(self, device: str, dim: int, scale: float = 0.25):
  
        mode_list = torch.tensor(
            [
                (a, b)
                for a in [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
                for b in [-7.0, -3.5, 0.0, 3.5, 7.0]
            ],
            device=device,
        )

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=scale,
        )
