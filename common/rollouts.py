import torch
from typing import Sequence
import numpy as np
from common.wrappers import *

from imitation.data import rollout, types
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm


def get_trajectories_probs(
        trans,
        policy,
) -> torch.Tensor:
    with torch.no_grad():
        log_probs = policy.get_log_prob_from_act(trans.obs, trans.acts)
    return log_probs


def generate_trajectories_from_data(
    data,
    env,
) -> Sequence[types.TrajectoryWithRew]:

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = rollout.TrajectoryAccumulator()
    trajectories_accum.add_step(dict(obs=data['state'][0]), 0)
    num_timesteps = 1
    if hasattr(env, "num_timesteps"):
        num_timesteps = env.num_timesteps
    obs_list = np.zeros([num_timesteps, env.observation_space.shape[0]])
    obs_list[0] = data['state'][0]
    acts_list = np.zeros([num_timesteps, env.action_space.shape[0]])
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
            obs_list = np.zeros([num_timesteps, env.observation_space.shape[0]])
            acts_list = np.zeros([num_timesteps, env.action_space.shape[0]])
        else:
            done = np.array([False])
            info["pltq"] = data['pltq'][i]
        rew = np.zeros([1, ])

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            act, obs, rew, done, [info]
        )
        trajectories.extend(new_trajs)

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


def generate_trajectories_without_shuffle(
    policy,
    venv: VecEnv,
    sample_until: rollout.GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
      policy (BasePolicy or BaseAlgorithm): A stable_baselines3 policy or algorithm
          trained on the gym environment.
      venv: The vectorized environments to interact with.
      sample_until: A function determining the termination condition.
          It takes a sequence of trajectories, and returns a bool.
          Most users will want to use one of `min_episodes` or `min_timesteps`.
      deterministic_policy: If True, asks policy to deterministically return
          action. Note the trajectories might still be non-deterministic if the
          environment has non-determinism!

    Returns:
      Sequence of trajectories, satisfying `sample_until`. Additional trajectories
      may be collected to avoid biasing process towards short episodes; the user
      should truncate if required.
    """
    get_action = policy.predict
    if isinstance(policy, BaseAlgorithm):
        policy.set_env(venv)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = rollout.TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=np.bool)
    while np.any(active):
        acts, _ = get_action(obs, deterministic=deterministic_policy)
        obs, rews, dones, infos = venv.step(acts)
        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts, obs, rews, dones, infos
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any environments
            # where a trajectory was completed this timestep.
            active &= ~dones

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