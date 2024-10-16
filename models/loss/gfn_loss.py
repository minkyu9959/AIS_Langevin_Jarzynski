import torch
from torch import Tensor


from models import GFN


def trajectory_balance_loss(
    log_prior: Tensor,
    log_pfs: Tensor,
    log_pbs: Tensor,
    log_Z: Tensor,
    log_reward: Tensor,
):
    loss = 0.5 * (
        (log_prior + log_pfs.sum(-1) + log_Z - log_pbs.sum(-1) - log_reward) ** 2
    )
    return loss.mean()


def annealed_db(
    log_prior: torch.Tensor,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    logZ_ratio: torch.Tensor,
    log_reward_t: torch.Tensor,
):
    loss = 0.5 * (
        (log_pfs + logZ_ratio + log_reward_t[:, :-1] - log_pbs - log_reward_t[:, 1:])
        ** 2
    ).sum(-1)

    return loss.mean()


def vargrad_loss(
    log_prior: Tensor,
    log_pfs: Tensor,
    log_pbs: Tensor,
    log_reward: Tensor,
):
    # Estimate log Z with batch average.
    log_Z = (log_reward + log_pbs.sum(-1) - log_pfs.sum(-1) - log_prior).mean(dim=0, keepdim=True) #Minkyu
    loss = log_Z + (log_prior + log_pfs.sum(-1) - log_reward - log_pbs.sum(-1))
    return 0.5 * (loss**2).mean()


def detailed_balance_loss(
    log_prior: Tensor,
    log_pfs: Tensor,
    log_pbs: Tensor,
    log_flows: Tensor,
    log_Z: Tensor,
    log_reward: Tensor,
):
    log_flows[:, -1] = log_reward

    loss = ((log_pfs + log_flows[:, :-1] - log_pbs - log_flows[:, 1:]) ** 2).sum(-1)
    loss += (log_Z + log_prior - log_flows[:, 0]) ** 2

    return (0.5 * loss).mean()


def mle_loss(log_prior: Tensor, log_pfs: torch.Tensor, log_pbs: torch.Tensor):

    loss = -(log_prior + log_pfs.sum(-1))

    return loss.mean()


def analyze_loss_fn_argument(loss_fn):
    if loss_fn is trajectory_balance_loss:
        need_log_Z = True
        need_log_reward = True
        need_log_flows = False
    elif loss_fn is vargrad_loss:
        need_log_Z = False
        need_log_reward = True
        need_log_flows = False
    elif loss_fn is detailed_balance_loss:
        need_log_Z = True
        need_log_reward = True
        need_log_flows = True
    elif loss_fn is mle_loss:
        need_log_Z = False
        need_log_reward = False
        need_log_flows = False
    elif loss_fn is annealed_db:
        need_log_Z = False
        need_log_reward = False
        need_log_flows = False
    else:
        raise Exception("Not supported loss type.")

    return need_log_Z, need_log_reward, need_log_flows


class GFNForwardLossWrapper:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.need_log_Z, self.need_log_reward, self.need_log_flows = (
            analyze_loss_fn_argument(loss_fn)
        )

        self.loss_is_annealed_db = loss_fn is annealed_db

    def __call__(
        self,
        gfn: GFN,
        batch_size: int,
        exploration_schedule=None,
        return_experience: bool = False,
    ):
        init_state = gfn.generate_initial_state(batch_size)
        log_prior = gfn.get_logprob_initial_state(init_state)

        traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
            init_state, exploration_schedule=exploration_schedule
        )

        kwargs = {
            "log_pfs": log_pfs,
            "log_pbs": log_pbs,
            "log_prior": log_prior,
        }

        if self.need_log_Z:
            kwargs["log_Z"] = gfn.learned_logZ

        if self.need_log_reward or return_experience:
            with torch.no_grad():
                log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

            if self.need_log_reward:
                kwargs["log_reward"] = log_reward

        if self.need_log_flows:
            kwargs["log_flows"] = gfn.get_flow_from_trajectory(traj)

        if self.loss_is_annealed_db:
            kwargs["logZ_ratio"] = gfn.logZ_ratio

            times = torch.linspace(0, 1, traj.size(1), device=traj.device)
            kwargs["log_reward_t"] = -gfn.annealed_energy.energy(times, traj)

        loss = self.loss_fn(**kwargs)

        if return_experience:
            return loss, traj, log_pfs, log_pbs, log_reward
        else:
            return loss


class GFNBackwardLossWrapper:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.need_log_Z, self.need_log_reward, self.need_log_flows = (
            analyze_loss_fn_argument(loss_fn)
        )

        self.loss_is_annealed_db = loss_fn is annealed_db

    def __call__(self, gfn: GFN, sample: torch.Tensor):
        traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
        log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

        kwargs = {
            "log_pfs": log_pfs,
            "log_pbs": log_pbs,
            "log_prior": log_prior,
        }

        if self.need_log_Z:
            kwargs["log_Z"] = gfn.learned_logZ

        if self.need_log_reward:
            kwargs["log_reward"] = gfn.energy_function.log_reward(sample)

        if self.need_log_flows:
            kwargs["log_flows"] = gfn.get_flow_from_trajectory(traj)

        if self.loss_is_annealed_db:
            kwargs["logZ_ratio"] = gfn.logZ_ratio

            times = torch.linspace(0, 1, traj.size(1), device=traj.device)
            kwargs["log_reward_t"] = -gfn.annealed_energy.energy(times, traj)

        return self.loss_fn(**kwargs)
