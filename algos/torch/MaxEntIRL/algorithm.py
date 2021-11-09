import os
from copy import deepcopy
from typing import List, Dict, Optional, Union, Tuple

import random
import numpy as np
import torch as th
from imitation.data.rollout import make_sample_until, flatten_trajectories
from imitation.data.types import Trajectory, Transitions
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
        # agent_obs = np.concatenate([agent_trans.obs, agent_trans.next_obs], axis=1)
        # expt_obs = np.concatenate([expt_trans.obs, expt_trans.next_obs], axis=1)
        agent_obs = agent_trans.obs
        expt_obs = expt_trans.obs
        if self.use_action_as_input:
            acts = agent_trans.acts
            if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                acts = self.wrap_eval_env.action(acts)
            agent_input = th.from_numpy(np.concatenate([agent_obs, acts], axis=1)).double()
            expt_input = th.from_numpy(np.concatenate([expt_obs, expt_trans.acts], axis=1)).double()
        else:
            agent_input = th.from_numpy(agent_obs).double()
            expt_input = th.from_numpy(expt_obs).double()
        agent_gammas = th.FloatTensor([self.agent.gamma ** i for i in range(len(agent_input))]).to(self.device)
        expt_gammas = th.FloatTensor([self.agent.gamma ** i for i in range(len(expt_input))]).to(self.device)
        agent_rewards = agent_gammas * self.reward_net(agent_input).flatten()
        expt_rewards = expt_gammas * self.reward_net(expt_input).flatten()
        return agent_rewards.sum(), expt_rewards.sum(), None

    def cal_loss(self, n_episodes) -> Tuple:
        expert_trajectories = deepcopy(self.expert_trajectories)
        agent_trajectories = random.sample(deepcopy(self.agent_trajectories), n_episodes * self.expand_ratio)
        target = th.cat([th.ones(len(expert_trajectories)), -th.ones(len(agent_trajectories))])
        trajs = expert_trajectories + agent_trajectories
        y = th.zeros(len(trajs))
        for i in range(0, len(trajs), 2):
            y[i], y[i+1], _ = self.mean_transition_reward(flatten_trajectories([trajs[i]]), flatten_trajectories([trajs[i+1]]))
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
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        agent_trajectories = generate_trajectories_without_shuffle(
                self.agent, self.vec_eval_env, sample_until, deterministic_policy=False)
        self.agent.set_env(self.wrap_env)
        agent_rewards, expert_rewards = 0, 0
        for i in range(n_episodes):
            agent_transition = flatten_trajectories([agent_trajectories[i]])
            expert_transition = flatten_trajectories([expert_trajectories[i]])
            agent_reward, expert_reward, _ = self.mean_transition_reward(agent_transition, expert_transition)
            agent_rewards += agent_reward.item()
            expert_rewards += expert_reward.item()
        loss = agent_rewards / n_episodes - expert_rewards / n_episodes
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
                reward_diffs = []
                for rew_steps in range(max_gradient_steps):
                    loss_diff, weight_norm, _ = self.cal_loss(n_episodes)
                    loss = loss_diff
                    self.reward_net.optimizer.zero_grad()
                    loss.backward()
                    with th.no_grad():
                        agents = flatten_trajectories(random.sample(deepcopy(self.agent_trajectories), n_episodes))
                        expts = flatten_trajectories(deepcopy(self.expert_trajectories))
                        agent_rewards, expt_rewards = 0, 0
                        for i in range(n_episodes):
                            agent_reward, expt_reward, _ = self.mean_transition_reward(agents, expts)
                            agent_rewards += agent_reward.item()
                            expt_rewards += expt_reward.item()
                        diffs = agent_rewards / n_episodes - expt_rewards / n_episodes
                        reward_diffs.append(diffs)
                    logger.record("weight norm", weight_norm)
                    logger.record("loss", loss_diff.item())
                    logger.record("reward_diff", diffs)
                    logger.record("steps", rew_steps, exclude="tensorboard")
                    logger.dump(rew_steps)
                    if np.mean(reward_diffs[-min_gradient_steps:]) <= -0.1 and \
                            reward_diffs[-1] < 0.0 and \
                            rew_steps >= min_gradient_steps:
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
                    logger.record("loss_diff", reward_diff)
                    logger.record("agent_steps", agent_steps, exclude="tensorboard")
                    logger.dump(agent_steps)
                    reward_diffs.append(reward_diff)
                    if np.mean(reward_diffs[-min_agent_iter:]) > 0 and agent_steps >= min_agent_iter and early_stop:
                        break 
            if callback:
                callback(self, itr)


class GuidedCostLearning(MaxEntIRL):
    def collect_rollouts(self, n_agent_episodes):
        self.agent_trajectories = []
        super().collect_rollouts(3*n_agent_episodes)

    def transition_is(self, transition: Transitions) -> Tuple[th.Tensor, th.Tensor]:
        if self.use_action_as_input:
            acts = transition.acts
            if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                acts = self.wrap_eval_env.action(acts)
            np_input = np.concatenate([transition.obs, acts], axis=1)
            th_input = th.from_numpy(np_input).double()
        else:
            th_input = th.from_numpy(transition.obs).double()
        gammas = th.FloatTensor([self.agent.gamma ** i for i in range(len(th_input))]).to(self.device)
        reward = th.sum(gammas * self.reward_net(th_input).flatten())
        log_prob = th.sum(get_trajectories_probs(transition, self.agent.policy))
        return reward, log_prob

    def cal_loss(self, n_episodes) -> Tuple:
        expert_transitions = flatten_trajectories(self.expert_trajectories)
        demo_trajs =  self.expert_trajectories + random.sample(self.agent_trajectories, n_episodes)
        islosses = th.zeros(len(demo_trajs)).double()
        log_probs = []
        for idx, traj in enumerate(demo_trajs):
            agent_transitions = flatten_trajectories([traj])
            reward, log_prob = self.transition_is(agent_transitions)
            islosses[idx] = reward - log_prob
            log_probs.append(log_prob.item())
        _, expt_reward, lcr = self.mean_transition_reward(agent_transitions, expert_transitions)
        loss = th.logsumexp(islosses, 0) - expt_reward / n_episodes
        logger.record("log_probs_mean", np.mean(log_probs))
        logger.record("log_probs_var", np.var(log_probs))
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
