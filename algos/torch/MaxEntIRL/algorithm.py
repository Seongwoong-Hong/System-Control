from copy import deepcopy
from typing import List, Dict, Optional, Union, Tuple

import random
import numpy as np
import torch as th
from imitation.data.rollout import make_sample_until, flatten_trajectories
from imitation.data.types import Transitions
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
            expert_transitions: Transitions,
            device='cpu',
            eval_env=None,
            rew_arch: List[int] = None,
            use_action_as_input: bool = True,
            rew_kwargs: Optional[Dict] = None,
            env_kwargs: Optional[Dict] = None,
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
        self.expert_transitions = expert_transitions
        self.agent_trajectories = []
        self.expand_ratio = 30

        if self.env_kwargs is None:
            self.env_kwargs = {}
        if self.eval_env is None:
            self.eval_env = env
        if self.rew_kwargs is None:
            self.rew_kwargs = {}

        num_timesteps = 1
        if hasattr(self.env, "num_timesteps"):
            num_timesteps = self.env.num_timesteps

        inp = self.env.observation_space.sample()
        if self.use_action_as_input:
            inp = np.concatenate([inp, self.env.action_space.sample()])
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
            ).double().to(self.device)
        elif RNet_type is "cnn":
            self.reward_net = CNNRewardNet(
                inp=inp.shape[1] * num_timesteps,
                arch=rew_arch,
                feature_fn=feature_fn,
                use_action_as_inp=self.use_action_as_input,
                device=self.device,
                **self.rew_kwargs
            ).double().to(self.device)
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
                self.agent, self.vec_eval_env, sample_until, deterministic_policy=False)
        self.agent.set_env(self.wrap_env)

    def mean_transition_reward(self, agent_trans: Transitions, expt_trans: Transitions) -> Tuple:
        if self.use_action_as_input:
            acts = agent_trans.acts
            if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                acts = self.wrap_eval_env.action(acts)
            agent_input = th.from_numpy(np.concatenate([agent_trans.obs, acts], axis=1)).double()
            expt_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1)).double()
        else:
            agent_input = th.from_numpy(agent_trans.obs).double()
            expt_input = th.from_numpy(expt_trans.obs).double()
        agent_rewards = self.reward_net(agent_input)
        expt_rewards = self.reward_net(expt_input)
        return agent_rewards.sum(), expt_rewards.sum(), None

    def cal_loss(self, n_episodes) -> Tuple:
        expert_transitions = deepcopy(self.expert_transitions)
        losses = []
        for i in range(0, len(self.agent_trajectories), self.expand_ratio * n_episodes):
            agent_transitions = flatten_trajectories(self.agent_trajectories[i:i + self.expand_ratio * n_episodes])
            agent_reward, expt_reward, _ = self.mean_transition_reward(agent_transitions, expert_transitions)
            losses.append(agent_reward / (self.expand_ratio * n_episodes) - expt_reward / n_episodes)
        weight_norm = 0.0
        for param in self.reward_net.parameters():
            weight_norm += param.norm().detach().item()
        loss = np.max(losses)
        return loss, weight_norm, None

    def sample_and_cal_loss(self, n_episodes):
        expert_transitions = deepcopy(self.expert_transitions)
        if isinstance(self.wrap_env, VecEnvWrapper):
            self.vec_eval_env = deepcopy(self.wrap_env)
            self.vec_eval_env.set_venv(DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(n_episodes)]))
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes * 5)
        agent_trajectories = generate_trajectories_without_shuffle(
                self.agent, self.vec_eval_env, sample_until, deterministic_policy=False)
        self.agent.set_env(self.wrap_env)
        agent_transitions = flatten_trajectories(agent_trajectories)
        agent_reward, expt_reward, _ = self.mean_transition_reward(agent_transitions, expert_transitions)
        loss = agent_reward / (n_episodes * 5) - expt_reward / n_episodes
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
            with logger.accumulate_means(f"{itr}/reward"):
                self.collect_rollouts(n_episodes)
                self.reward_net.train()
                losses = []
                for rew_steps in range(max_gradient_steps):
                    loss_diff, weight_norm, _ = self.cal_loss(n_episodes)
                    loss = loss_diff
                    self.reward_net.optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss_diff.item())
                    logger.record("weight norm", weight_norm)
                    logger.record("loss", loss_diff.item())
                    logger.record("steps", rew_steps, exclude="tensorboard")
                    logger.dump(rew_steps)
                    if np.mean(losses[-min_gradient_steps:]) <= 0 and rew_steps >= min_gradient_steps:
                        self.reward_net.optimizer.zero_grad()
                        break
                    self.reward_net.optimizer.step()
            with logger.accumulate_means(f"{itr}/agent"):
                self._reset_agent(n_episodes, **self.env_kwargs)
                loss_diffs = []
                for agent_steps in range(1, max_agent_iter + 1):
                    self.agent.learn(
                        total_timesteps=int(agent_learning_steps), reset_num_timesteps=False, callback=agent_callback)
                    logger.dump(step=self.agent.num_timesteps)
                    loss_diff = self.sample_and_cal_loss(n_episodes)
                    logger.record("loss_diff", loss_diff.item())
                    logger.record("agent_steps", agent_steps, exclude="tensorboard")
                    logger.dump(agent_steps)
                    loss_diffs.append(loss_diff.item())
                    if np.mean(loss_diffs[-min_agent_iter:]) > 0 and agent_steps >= min_agent_iter and early_stop:
                        break 
            if callback:
                callback(self, itr)


class GuidedCostLearning(MaxEntIRL):
    def transition_is(self, transition: Transitions) -> th.Tensor:
        if self.use_action_as_input:
            acts = transition.acts
            if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                acts = self.wrap_eval_env.action(acts)
            np_input = np.concatenate([transition.obs, acts], axis=1)
            th_input = th.from_numpy(np_input).double()
        else:
            th_input = th.from_numpy(transition.obs).double()
        costs = -th.sum(self.reward_net(th_input))
        log_probs = th.sum(get_trajectories_probs(transition, self.agent.policy))
        return -costs + log_probs

    def cal_loss(self, **kwargs) -> Tuple:
        n_episodes = kwargs.pop('n_episodes', 10)
        n_agent_episodes = n_episodes * self.vec_eval_env.num_envs
        expert_transitions = deepcopy(self.expert_transitions)
        agent_trajs = self.rollout_from_agent(n_agent_episodes)
        losses = th.zeros(len(agent_trajs))
        for i, traj in enumerate(agent_trajs):
            losses[i] = self.transition_is(flatten_trajectories([traj]))
        IOCLoss2 = th.logsumexp(losses, 0)
        agent_transitions = flatten_trajectories(agent_trajs)
        _, expt_reward, lcr = self.mean_transition_reward(agent_transitions, expert_transitions)
        weight_norm = 0.0
        for param in self.reward_net.parameters():
            weight_norm += param.norm().detach().item()
        return -expt_reward / n_episodes + IOCLoss2, weight_norm, lcr

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
