import copy
import torch
from typing import Sequence

import numpy as np

from imitation.data import rollout
from stable_baselines3.common.vec_env import VecEnv
from imitation.data import types


def get_trajectories_probs(
        trans,
        policy,
) -> torch.Tensor:
    obs = torch.from_numpy(trans[0]['obs'].reshape(1, -1)).to(policy.device)
    acts = torch.from_numpy(trans[0]['acts'].reshape(1, -1)).to(policy.device)
    log_probs = policy.get_log_prob_from_act(obs, acts)
    for i in range(len(trans)-1):
        obs = torch.from_numpy(trans[i+1]['obs'].reshape(1, -1)).to(policy.device)
        acts = torch.from_numpy(trans[i+1]['acts'].reshape(1, -1)).to(policy.device)
        log_probs = torch.cat([log_probs, policy.get_log_prob_from_act(obs, acts)], dim=0)
    return log_probs


def generate_trajectories_from_data(
    data,
    env,
    *,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = rollout.TrajectoryAccumulator()
    trajectories_accum.add_step(dict(obs=data['state'][0]), 0)
    obs_list = np.zeros([env.num_timesteps, env.observation_space.shape[0]])
    obs_list[0] = data['state'][0]
    acts_list = np.zeros([env.num_timesteps, env.action_space.shape[0]])
    for i in range(len(data['state']) - 1):
        obs = data['state'][i + 1].reshape(1, -1)
        act = data['T'][i].reshape(1, -1)
        acts_list = np.append(act, acts_list[:-1], axis=0)
        info = {'obs': obs_list, 'acts': acts_list}
        obs_list = np.append(obs, obs_list[:-1], axis=0)
        if i + 1 == len(data['state']) - 1:
            # Termination condition has been reached. Mark as inactive any environments
            # where a trajectory was completed this timestep.
            done = np.array([True])
            info["terminal_observation"] = data['state'][i]
            info["pltq"] = data["pltq"][i]
            obs_list = np.zeros([env.num_timesteps, env.observation_space.shape[0]])
            acts_list = np.zeros([env.num_timesteps, env.action_space.shape[0]])
        else:
            done = np.array([False])
            info["pltq"] = data['pltq'][i]
        rew = np.zeros([1, ])

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            act, obs, rew, done, [info]
        )
        trajectories.extend(new_trajs)

    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + env.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + env.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories
