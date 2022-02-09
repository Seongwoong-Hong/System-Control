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
    for i in range(len(data['state'])):
        if i >= (len(data['state']) - 1):
            # Termination condition has been reached. Mark as inactive any environments
            # where a trajectory was completed this timestep.
            obs = data['state'][-1].reshape(1, -1)
            act = data['T'][-1].reshape(1, -1)
            done = np.array([True])
            info = {"terminal_observation": data['state'][-1]}
        else:
            act = data['T'][i].reshape(1, -1)
            obs = data['state'][i + 1].reshape(1, -1)
            done = np.array([False])
            info = {}
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


class DiscEnvTrajectories:
    def __init__(self):
        self.rews = None
        self.obs = None
        self.acts = None


def generate_trajectories_from_approx_dyn(agent, env, n_episodes, deterministic_policy=False):
    trajs = []
    P = env.env_method("get_trans_mat")[0]
    for _ in range(n_episodes):
        traj = DiscEnvTrajectories()
        obs_approx, rews_approx, acts_approx = [], [], []
        s_vec, _ = env.env_method("get_vectorized")[0]
        current_obs = env.reset()[0]
        obs_approx.append(current_obs)
        for _ in range(env.get_attr("spec")[0].max_episode_steps):
            act, _ = agent.predict(current_obs, deterministic=deterministic_policy)
            rews_approx.append(env.env_method("get_reward", current_obs, act)[0])
            acts_approx.append(act)
            torque = env.env_method("get_torque", act)[0].T
            a_ind = env.env_method("get_idx_from_acts", torque)[0]
            obs_ind = env.env_method("get_ind_from_state", current_obs.squeeze())[0]
            next_obs = obs_ind @ P[a_ind[0]].T @ s_vec
            obs_approx.append(next_obs)
            current_obs = next_obs
        traj.obs = np.vstack(obs_approx)
        traj.acts = np.vstack(acts_approx)
        traj.rews = np.array(rews_approx).flatten()
        trajs.append(traj)
    return trajs
