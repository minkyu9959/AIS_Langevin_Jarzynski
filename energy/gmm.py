from typing import Optional

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .base_energy import BaseEnergy
from .utils import draw_2D_contour, draw_2D_sample


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
        plotting_bounds: tuple = (-10.0, 10.0),
    ):
        assert dim == 2
        super().__init__(device=device, dim=dim)

        self.plotting_bounds = plotting_bounds

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

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.gmm.log_prob(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.gmm.log_prob(x)

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return self.gmm.sample((batch_size,))

    def plot_contours(
        self,
        ax: Axes,
        grid_width_n_points: int = 200,
        n_contour_levels: int = 50,
        log_prob_min: float = -1000.0,
    ):
        """
        Plot contours of a log_prob func that is defined on 2D.
        This function returns contour object.
        """
        return draw_2D_contour(
            ax,
            log_prob_func=self.log_prob,
            bounds=self.plotting_bounds,
            device=self.device,
            grid_width_n_points=grid_width_n_points,
            n_contour_levels=n_contour_levels,
            log_prob_min=log_prob_min,
        )

    def plot_sample(self, sample: torch.Tensor, ax: Axes, alpha: float = 0.5):
        """
        Draw sample plot on 2D.
        This function returns scatter object.
        """
        return draw_2D_sample(sample, ax, self.plotting_bounds, alpha)

    def plot_ground_truth_sample(self, ax: Axes, alpha: float = 0.5):
        """
        Draw ground truth sample plot on 2D.
        This function returns scatter object.
        """
        return self.plot_sample(self.sample(batch_size=10000), ax, alpha)

    def make_plot(self, sample: torch.Tensor):
        """
        Make figure containing sample from the model.

        Args:
            sample (torch.Tensor): Sample generated by model.

        Return:
            fig, axs: matplotlib figure and axes object that is created.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))

        self.plot_contours(axs[0])
        self.plot_ground_truth_sample(axs[0])
        axs[0].set_title("Ground truth sample")

        self.plot_contours(axs[1])
        self.plot_sample(sample, axs[1])
        axs[1].set_title("Sample from the model")

        return fig, axs


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
            plotting_bounds=(-8.0, 8.0),
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
            plotting_bounds=(-15.0, 15.0),
        )


class GMM40(GaussianMixture):
    pass
