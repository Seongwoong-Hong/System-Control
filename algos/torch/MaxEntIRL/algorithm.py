import os
from copy import deepcopy
from typing import List, Dict, Optional, Union, Tuple, Sequence

import random
import numpy as np
import torch as th
from imitation.data.rollout import make_sample_until, flatten_trajectories
from imitation.data.types import Trajectory, Transitions, TrajectoryWithRew
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.monitor import Monitor

from algos.torch.MaxEntIRL import RewardNet, CNNRewardNet
from common.wrappers import ActionRewardWrapper
from common.rollouts import get_trajectories_probs, generate_trajectories_without_shuffle


class MaxEntIRL:
    def __init__(
            self,
            env: GymEnv,
            feature_fn,
            agent,
            expert_trajectories: List[Trajectory],
            device: str = 'cpu',
            eval_env: GymEnv = None,
            rew_arch: List[int] = None,
            use_action_as_input: bool = True,
            rew_kwargs: Optional[Dict] = None,
            env_kwargs: Optional[Dict] = None,
            **kwargs,
    ):
        assert (
            logger.is_configured()
        ), "Requires calling imitation.util.logger.configure"

        self.env = env
        self.agent = agent
        self.device = device
        self.use_action_as_input = use_action_as_input
        self.env_kwargs = env_kwargs
        self.rew_kwargs = rew_kwargs
        self.eval_env = eval_env
        self.expert_trajectories = expert_trajectories
        self.expert_transitions = flatten_trajectories(expert_trajectories)
        self.agent_trajectories = []
        self.expand_ratio = 11

        if self.env_kwargs is None:
            self.env_kwargs = {}
        if self.eval_env is None:
            self.eval_env = env
        if self.rew_kwargs is None:
            self.rew_kwargs = {}

        num_timesteps = 1
        if hasattr(self.env, "num_timesteps"):
            num_timesteps = self.env.num_timesteps

        self.env.reset()
        inp, _, _, _ = self.env.step(self.env.action_space.sample())
        if self.use_action_as_input:
            inp = np.append(inp, self.env.action_space.sample())
        inp = feature_fn(th.from_numpy(inp).reshape(1, -1))

        RNet_type = rew_kwargs.pop("type", None)
        if RNet_type is None or RNet_type is "ann":
            self.reward_net = RewardNet(
                inp=inp.shape[1] * num_timesteps,
                arch=rew_arch,
                feature_fn=feature_fn,
                use_action_as_inp=self.use_action_as_input,
                device=self.device,
                **self.rew_kwargs
            ).to(self.device)
        elif RNet_type is "cnn":
            self.reward_net = CNNRewardNet(
                inp=inp.shape[1] * num_timesteps,
                arch=rew_arch,
                feature_fn=feature_fn,
                use_action_as_inp=self.use_action_as_input,
                device=self.device,
                **self.rew_kwargs
            ).to(self.device)
        else:
            raise NotImplementedError("Not implemented reward net type")

    def _reset_agent(self, n_agent_episodes, **kwargs):
        reward_wrapper = kwargs.pop("reward_wrapper", ActionRewardWrapper)
        norm_wrapper = kwargs.pop("vec_normalizer", None)
        self.reward_net.eval()
        self.wrap_env = reward_wrapper(deepcopy(self.env), self.reward_net.eval())
        self.wrap_eval_env = reward_wrapper(self.eval_env, self.reward_net.eval())
        if norm_wrapper:
            self.wrap_env = norm_wrapper(DummyVecEnv([lambda: Monitor(self.wrap_env)]), **kwargs)
        else:
            self.wrap_env = DummyVecEnv([lambda: Monitor(self.wrap_env)])
            self.vec_eval_env = DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(n_agent_episodes)])
        self.agent.reset(self.wrap_env)

    def collect_rollouts(self, n_agent_episodes):
        print("Collecting rollouts from the current agent...")
        if isinstance(self.wrap_env, VecEnvWrapper):
            self.vec_eval_env = deepcopy(self.wrap_env)
            self.vec_eval_env.set_venv(DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(n_agent_episodes)]))
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_agent_episodes * self.expand_ratio)
        self.agent_trajectories += generate_trajectories_without_shuffle(
                self.agent, self.vec_eval_env, sample_until, deterministic_policy=True)
        self.agent.set_env(self.wrap_env)

    def mean_transition_reward(self, agent_trajs: Sequence[Trajectory], expt_trajs: Sequence[Trajectory]) -> Tuple:
        assert len(agent_trajs[0].acts) == len(expt_trajs[0].acts)
        traj_len = len(expt_trajs[0].acts)
        # agent_obs = np.concatenate([agent_trans.obs, agent_trans.next_obs], axis=1)
        # expt_obs = np.concatenate([expt_trans.obs, expt_trans.next_obs], axis=1)
        agent_trans = flatten_trajectories(agent_trajs)
        expt_trans = flatten_trajectories(expt_trajs)
        agent_obs = agent_trans.obs
        expt_obs = expt_trans.obs
        if self.use_action_as_input:
            acts = agent_trans.acts
            if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                acts = self.wrap_eval_env.action(acts)
            agent_input = th.from_numpy(np.concatenate([agent_obs, acts], axis=1)).float()
            expt_input = th.from_numpy(np.concatenate([expt_obs, expt_trans.acts], axis=1)).float()
        else:
            agent_input = th.from_numpy(agent_obs).float()
            expt_input = th.from_numpy(expt_obs).float()
        agent_gammas = th.FloatTensor([self.agent.gamma ** (i % traj_len) for i in range(len(agent_trans))]).to(self.device)
        expt_gammas = th.FloatTensor([self.agent.gamma ** (i % traj_len) for i in range(len(expt_trans))]).to(self.device)
        trans_agent_rewards = agent_gammas * self.reward_net(agent_input).flatten()
        trans_expt_rewards = expt_gammas * self.reward_net(expt_input).flatten()
        agent_rewards, expt_rewards = th.zeros(len(agent_trajs)), th.zeros(len(expt_trajs))
        for i in range(len(agent_trajs)):
            agent_rewards[i] = trans_agent_rewards[i*traj_len:(i+1)*traj_len].sum()
        for i in range(len(expt_trajs)):
            expt_rewards[i] = trans_expt_rewards[i*traj_len:(i+1)*traj_len].sum()
        return agent_rewards, expt_rewards, None

    def cal_loss(self, n_episodes) -> Tuple:
        expert_trajectories = deepcopy(self.expert_trajectories)
        agent_trajectories = deepcopy(self.current_agent_trajectories)
        target = th.cat([th.ones(len(expert_trajectories)), -th.ones(len(agent_trajectories))])
        agent_rewards, expert_rewards, _ = self.mean_transition_reward(agent_trajectories, expert_trajectories)
        y = th.cat([expert_rewards, agent_rewards], dim=0)
        loss = th.mean(th.clamp(1 - y * target, min=0))
        weight_norm = 0.0
        for param in self.reward_net.parameters():
            weight_norm += param.norm().detach().item()
        return loss, weight_norm, None

    def sample_and_cal_loss(self, n_episodes):
        expert_trajectories = deepcopy(self.expert_trajectories)
        if isinstance(self.wrap_env, VecEnvWrapper):
            self.vec_eval_env = deepcopy(self.wrap_env)
            self.vec_eval_env.set_venv(DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(n_episodes)]))
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes*self.env.observation_space.nvec[0])
        agent_trajectories = generate_trajectories_without_shuffle(
                self.agent, self.vec_eval_env, sample_until, deterministic_policy=True)
        self.agent.set_env(self.wrap_env)
        agent_rewards, expert_rewards, _ = self.mean_transition_reward(agent_trajectories, expert_trajectories)
        loss = th.mean(agent_rewards) - th.mean(expert_rewards)
        return loss

    def learn(
            self,
            total_iter: int,
            agent_learning_steps: Union[int, float],
            max_gradient_steps: int = 40,
            min_gradient_steps: int = 15,
            max_agent_iter: int = 15,
            min_agent_iter: int = 3,
            agent_callback=None,
            callback=None,
            early_stop=True,
            n_episodes: int = 10,
            **kwargs
    ):
        self._reset_agent(n_episodes, **self.env_kwargs)
        for itr in range(total_iter):
            self.reward_net.reset_optim()
            with logger.accumulate_means(f"{itr}/reward"):
                self.collect_rollouts(n_episodes)
                self.reward_net.train()
                reward_diffs = []
                for rew_steps in range(max_gradient_steps):
                    self.current_agent_trajectories = random.sample(self.agent_trajectories, n_episodes)
                    loss_diff, weight_norm, _ = self.cal_loss(n_episodes)
                    loss = loss_diff
                    self.reward_net.optimizer.zero_grad()
                    loss.backward()
                    with th.no_grad():
                        agents = deepcopy(self.current_agent_trajectories)
                        expts = deepcopy(self.expert_trajectories)
                        agent_rewards, expt_rewards, _ = self.mean_transition_reward(agents, expts)
                        diffs = agent_rewards.mean().item() - expt_rewards.mean().item()
                        reward_diffs.append(diffs)
                    logger.record("weight norm", weight_norm)
                    logger.record("loss", loss_diff.item())
                    logger.record("reward_diff", diffs)
                    logger.record("steps", rew_steps, exclude="tensorboard")
                    logger.dump(rew_steps)
                    if np.mean(reward_diffs[-min_gradient_steps:]) <= -0.1 and \
                            reward_diffs[-1] < 0.0 and \
                            rew_steps >= min_gradient_steps and early_stop:
                        self.reward_net.optimizer.zero_grad()
                        break
                    self.reward_net.optimizer.step()
            with logger.accumulate_means(f"{itr}/agent"):
                self._reset_agent(n_episodes, **self.env_kwargs)
                reward_diffs = []
                for agent_steps in range(1, max_agent_iter + 1):
                    self.agent.learn(
                        total_timesteps=int(agent_learning_steps), reset_num_timesteps=False, callback=agent_callback)
                    logger.dump(step=self.agent.num_timesteps)
                    reward_diff = self.sample_and_cal_loss(n_episodes)
                    logger.record("loss_diff", reward_diff.item())
                    logger.record("agent_steps", agent_steps, exclude="tensorboard")
                    logger.dump(agent_steps)
                    reward_diffs.append(reward_diff.item())
                    if np.mean(reward_diffs[-min_agent_iter:]) > 0 and \
                            reward_diffs[-1] > 0.0 and \
                            agent_steps >= min_agent_iter and early_stop:
                        break 
            if callback:
                callback(self, itr)


class GuidedCostLearning(MaxEntIRL):
    def collect_rollouts(self, n_agent_episodes):
        self.agent_trajectories = []
        super().collect_rollouts(3*n_agent_episodes)

    def transition_is(self, trajectories: Sequence[Trajectory]) -> Tuple[th.Tensor, th.Tensor]:
        traj_len = len(trajectories[0].acts)
        transitions = flatten_trajectories(trajectories)
        if self.use_action_as_input:
            acts = transitions.acts
            if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                acts = self.wrap_eval_env.action(acts)
            np_input = np.concatenate([transitions.obs, acts], axis=1)
            th_input = th.from_numpy(np_input)
        else:
            th_input = th.from_numpy(transitions.obs)
        gammas = th.FloatTensor([self.agent.gamma ** (i % traj_len) for i in range(len(th_input))]).to(self.device)
        trans_reward = gammas * self.reward_net(th_input).flatten()
        trans_log_prob = get_trajectories_probs(transitions, self.agent.policy)
        rewards, log_probs = th.zeros(len(trajectories)), th.zeros(len(trajectories))
        for i in range(len(trajectories)):
            rewards[i] = trans_reward[i*traj_len:(i+1)*traj_len].sum()
            log_probs[i] = trans_log_prob[i*traj_len:(i+1)*traj_len].sum()
        return rewards, log_probs

    def cal_loss(self, n_episodes) -> Tuple:
        expert_trajectories = deepcopy(self.expert_trajectories)
        demo_trajs = expert_trajectories + deepcopy(self.current_agent_trajectories)
        rewards, log_probs = self.transition_is(demo_trajs)
        _, expt_rewards, lcr = self.mean_transition_reward([demo_trajs[0]], expert_trajectories)
        loss = th.logsumexp(rewards - log_probs, 0) - expt_rewards.mean()
        logger.record("log_probs_mean", th.mean(log_probs).item())
        logger.record("log_probs_var", th.var(log_probs).item())
        weight_norm = 0.0
        for param in self.reward_net.parameters():
            weight_norm += param.norm().detach().item()
        return loss, weight_norm, None

    def learn(
            self,
            total_iter: int,
            agent_learning_steps: Union[int, float],
            max_gradient_steps: int = 40,
            min_gradient_steps: int = 15,
            max_agent_iter: int = 15,
            min_agent_iter: int = 3,
            agent_callback=None,
            callback=None,
            early_stop=False,
            **kwargs
    ):
        super(GuidedCostLearning, self).learn(
            total_iter=total_iter,
            agent_learning_steps=agent_learning_steps,
            max_gradient_steps=max_gradient_steps,
            min_gradient_steps=min_gradient_steps,
            max_agent_iter=max_agent_iter,
            min_agent_iter=min_agent_iter,
            agent_callback=agent_callback,
            callback=callback,
            early_stop=early_stop,
            **kwargs,
        )
