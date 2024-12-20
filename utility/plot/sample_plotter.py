from typing import Optional

import torch

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation

from .draw_plot import *

from energy import BaseEnergy, HighDimensionalEnergy, AnnealedDensities


class SamplePlotter:
    def __init__(
        self,
        energy_function: BaseEnergy,
        plotting_bounds=(-10.0, 10.0),
        projection_dims: Optional[list[tuple[int, int]]] = None,
        fig_size: Optional[tuple[float, float]] = (12.0, 6.0),
        **kwargs,
    ):
        self.energy_function = energy_function
        self.need_projection = isinstance(energy_function, HighDimensionalEnergy)

        if self.need_projection and projection_dims is None:
            raise Exception("Please provide projection_dims for HighDimensionalEnergy")

        self.projection_dims = projection_dims

        # Figure settings
        self.fig_size = fig_size

        self.plotting_bounds = plotting_bounds

        self.grid_width_n_points = 200
        self.n_contour_levels = 50
        self.log_prob_min = -1000.0

        self.alpha = 0.3

    def draw_contour(
        self,
        ax: Axes,
        first_dim: Optional[int] = None,
        second_dim: Optional[int] = None,
    ):
        if self.need_projection:

            def log_prob_2D(x_2D: torch.Tensor) -> torch.Tensor:
                return -self.energy_function.energy_on_2d(x_2D, first_dim, second_dim)

            log_prob_func = log_prob_2D
        else:

            def log_prob(x: torch.Tensor) -> torch.Tensor:
                return -self.energy_function.energy(x)

            log_prob_func = log_prob

        return draw_2D_contour(
            ax,
            log_prob_func=log_prob_func,
            bounds=self.plotting_bounds,
            device=self.energy_function.device,
            grid_width_n_points=self.grid_width_n_points,
            n_contour_levels=self.n_contour_levels,
            log_prob_min=self.log_prob_min,
        )

    def draw_sample(
        self,
        sample: torch.Tensor,
        ax: Axes,
        first_dim: Optional[int] = None,
        second_dim: Optional[int] = None,
    ):
        """
        Make figure containing sample from the model.

        Args:
            sample (torch.Tensor): Sample generated by model.

        Return:
            fig, axs: matplotlib figure and axes object that is created.
        """
        if self.need_projection:
            sample = self.energy_function.projection_on_2d(
                sample, first_dim, second_dim
            )

        return draw_2D_sample(
            sample,
            ax,
            self.plotting_bounds,
            self.alpha,
        )

    def draw_contour_and_ground_truth_sample(
        self,
        ax: Axes,
        first_dim: Optional[int] = None,
        second_dim: Optional[int] = None,
    ):
        ground_truth_sample = self.energy_function.sample(batch_size=2000)
        self.draw_contour_and_sample(ground_truth_sample, ax, first_dim, second_dim)

        ax.set_title("Ground truth sample plot")

    def draw_contour_and_sample(
        self,
        sample: torch.Tensor,
        ax: Axes,
        first_dim: Optional[int] = None,
        second_dim: Optional[int] = None,
    ):
        self.draw_contour(ax, first_dim, second_dim)
        self.draw_sample(sample, ax, first_dim, second_dim)

        if self.need_projection:
            ax.set_title(f"Projected on x{first_dim}, x{second_dim}")

            ax.set_ylabel(f"x{second_dim}")
            ax.set_xlabel(f"x{first_dim}")
        else:
            ax.set_title(f"Sample plot")

    def make_trajectory_plot(
        self,
        trajectory: torch.Tensor,
        first_dim: Optional[int] = None,
        second_dim: Optional[int] = None,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        self.draw_contour(ax, first_dim, second_dim)

        if self.need_projection:
            energy: HighDimensionalEnergy = self.energy_function
            trajectory = energy.projection_on_2d(trajectory, first_dim, second_dim)

        draw_sample_trajectory_plot(ax, trajectory)

        return fig, ax

    def make_sample_generation_animation(
        self,
        trajectory: torch.Tensor,
        first_dim: Optional[int] = None,
        second_dim: Optional[int] = None,
    ):
        """
        Make animation describing sample generation for given trajectory.

        Args:
            trajectory (torch.Tensor): sample generation trajectory from model.

        Return:
            animation, fig, axs: matplotlib animation, figure and axes object that is created.
        """
        trajectory_length = trajectory.size(1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        self.draw_contour(ax, first_dim, second_dim)
        scatter_obj = self.draw_sample(trajectory[:, 0, :], ax, first_dim, second_dim)

        trajectory = trajectory.cpu().numpy()

        def update(frame: int):
            scatter_obj.set_offsets(trajectory[:, frame, :])

        animation = FuncAnimation(fig, update, frames=trajectory_length, interval=200)

        return animation, fig, ax

    def make_sample_plot(self, sample: torch.Tensor):
        """
        Make figure containing sample from the model.

        Args:
            sample (torch.Tensor): Sample generated by model.

        Return:
            fig, axs: matplotlib figure and axes objec
            t that is created.
        """

        # Energy function is defined on high dimensional space.
        if self.need_projection:
            nrows = (len(self.projection_dims) + 1) // 2
            ncols = 2

            fig, axs = plt.subplots(nrows, ncols, figsize=self.fig_size)

            for (proj_dim_1, proj_dim_2), ax in zip(
                self.projection_dims, axs.flatten()
            ):
                self.draw_contour_and_sample(sample, ax, proj_dim_1, proj_dim_2)

        # Energy function is 2D.
        else:
            fig, axs = plt.subplots(1, 1, figsize=self.fig_size)
            self.draw_contour_and_sample(sample, axs)

        return fig, axs

    def make_kde_plot(self, sample: torch.Tensor):
        """KDE plot for sample."""

        # Energy function is defined on high dimensional space.
        if self.need_projection:
            nrows = (len(self.projection_dims) + 1) // 2
            ncols = 2

            fig, axs = plt.subplots(nrows, ncols, figsize=self.fig_size)

            for (proj_dim_1, proj_dim_2), ax in zip(
                self.projection_dims, axs.flatten()
            ):
                proj_sample = self.energy_function.projection_on_2d(
                    sample, proj_dim_1, proj_dim_2
                )
                draw_2D_kde(proj_sample, ax, self.plotting_bounds)

        # Energy function is 2D.
        else:
            fig, axs = plt.subplots(1, 1, figsize=self.fig_size)
            draw_2D_kde(sample, axs, self.plotting_bounds)

        return fig, axs

    def make_time_logZ_plot(
        self, annealed_density: AnnealedDensities, logZ_ratio: torch.Tensor
    ):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        logZ_t = annealed_density.logZ_t(100000, logZ_ratio.size(0))
        draw_time_logZ_plot(ax, logZ_t)

        learned_logZ_t = (
            logZ_ratio.detach().cumsum(dim=0)
            + annealed_density.prior_energy.ground_truth_logZ
        )
        draw_time_logZ_plot(ax, learned_logZ_t, label="Learned logZ_t")

        ax.legend()

        return fig, ax

    def make_energy_histogram(self, sample: torch.Tensor, name: str = ""):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        log_reward = self.energy_function.log_reward(sample)

        draw_energy_histogram(ax, log_reward, name)

        return fig, ax

    # def make_histogram_plot(self, data, bins=50, range=(-5, 5)):
    #     fig, ax = plt.subplots(figsize=self.fig_size)
    #     ax.hist(data, bins=bins, range=range)
    #     return fig, ax