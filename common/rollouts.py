import copy
import torch
from typing import Sequence

import numpy as np

from imitation.data import rollout
from stable_baselines3.common.vec_env import VecEnv
from imitation.data import types


def get_trajectories_probs(
        trajectories,
        policy,
) -> Sequence[torch.Tensor]:
    transitions = []
    for traj in trajectories:
        trans = copy.deepcopy(rollout.flatten_trajectories_with_rew([traj]))
        obs = torch.from_numpy(trans[0]['obs'].reshape(1, -1)).to(policy.device)
        acts = torch.from_numpy(trans[0]['acts'].reshape(1, -1)).to(policy.device)
        log_probs = policy.get_log_prob_from_act(obs, acts)
        concats = torch.cat((obs, acts, log_probs.reshape(-1, 1)), dim=1)
        for i in range(len(trans)-1):
            obs = torch.from_numpy(trans[i+1]['obs'].reshape(1, -1)).to(policy.device)
            acts = torch.from_numpy(trans[i+1]['acts'].reshape(1, -1)).to(policy.device)
            log_probs = policy.get_log_prob_from_act(obs, acts)
            concats = torch.cat((concats, torch.cat((obs, acts, log_probs.reshape(-1, 1)), dim=1)), dim=0)
        transitions += [concats]
    return transitions


def generate_trajectories_from_data(
    data,
    venv: VecEnv,
    *,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = rollout.TrajectoryAccumulator()
    trajectories_accum.add_step(dict(obs=data['state'][0]), 0)
    for i in range(len(data['state'])):
        act = data['T'][i].reshape(1, -1)
        if (i + 1) == len(data['state']):
            # Termination condition has been reached. Mark as inactive any environments
            # where a trajectory was completed this timestep.
            done = np.array([True])
            obs = data['state'][i].reshape(1, -1)
            info = [{"terminal_observation": data['state'][i], "pltq": data["pltq"][i]}]
        else:
            done = np.array([False])
            obs = data['state'][i+1].reshape(1, -1)
            info = [{"pltq": data['pltq'][i]}]
        rew = np.zeros([1, ])

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            act, obs, rew, done, info
        )
        trajectories.extend(new_trajs)

    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories
