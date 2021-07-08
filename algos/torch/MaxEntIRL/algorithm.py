from copy import deepcopy
from typing import List, Dict, Optional, Union

import numpy as np
import torch as th
from imitation.data.rollout import make_sample_until, generate_trajectories, flatten_trajectories
from imitation.data.types import Transitions
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.type_aliases import GymEnv

from algos.torch.MaxEntIRL import RewardNet, CNNRewardNet
from common.wrappers import RewardWrapper
from common.rollouts import get_trajectories_probs


class MaxEntIRL:
    def __init__(
            self,
            env: GymEnv,
            feature_fn,
            agent,
            expert_transitions: Transitions,
            device='cpu',
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
        if env_kwargs is None:
            self.env_kwargs = {}
        else:
            self.env_kwargs = env_kwargs
        if rew_kwargs is None:
            self.rew_kwargs = {}
        else:
            self.rew_kwargs = rew_kwargs
        self.expert_transitions = expert_transitions
        inp = self.env.observation_space.sample()
        if self.use_action_as_input:
            inp = np.concatenate([inp, self.env.action_space.sample()])
        inp = feature_fn(th.from_numpy(inp).reshape(1, -1))
        RNet_type = rew_kwargs.pop("type", None)
        if RNet_type is None or RNet_type is "ann":
            self.reward_net = RewardNet(
                inp=inp.shape[1],
                arch=rew_arch,
                feature_fn=feature_fn,
                use_action_as_inp=self.use_action_as_input,
                device=self.device,
                **self.rew_kwargs
            ).double().to(self.device)
        elif RNet_type is "cnn":
            self.reward_net = CNNRewardNet(
                inp=inp.shape[1],
                arch=rew_arch,
                feature_fn=feature_fn,
                use_action_as_inp=self.use_action_as_input,
                device=self.device,
                **self.rew_kwargs
            ).double().to(self.device)
        else:
            raise NotImplementedError("Not implemented reward net type")

    def _reset_agent(self, **kwargs):
        reward_wrapper = kwargs.pop("reward_wrapper", RewardWrapper)
        norm_wrapper = kwargs.pop("vec_normalizer", None)
        self.reward_net.eval()
        if norm_wrapper:
            self.wrap_env = norm_wrapper(DummyVecEnv([lambda: reward_wrapper(self.env, self.reward_net)]), **kwargs)
        else:
            self.wrap_env = DummyVecEnv([lambda: reward_wrapper(self.env, self.reward_net)])
        self.agent.reset(self.wrap_env)

    def rollout_from_agent(self, **kwargs):
        n_episodes = kwargs.pop('n_episodes', 10)
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        trajectories = generate_trajectories(
            self.agent, self.wrap_env, sample_until, deterministic_policy=False)
        return trajectories

    def mean_transition_reward(self, transition: Transitions) -> th.Tensor:
        if self.use_action_as_input:
            np_input = np.concatenate([transition.obs, transition.acts], axis=1)
            th_input = th.from_numpy(np_input).double()
        else:
            th_input = th.from_numpy(transition.obs).double()
        reward = self.reward_net(th_input)
        return reward.mean()

    def cal_loss(self, **kwargs) -> th.Tensor:
        expert_transitions = deepcopy(self.expert_transitions)
        agent_transitions = flatten_trajectories(self.rollout_from_agent(**kwargs))
        agent_ex = self.mean_transition_reward(agent_transitions)
        expert_ex = self.mean_transition_reward(expert_transitions)
        loss = agent_ex - expert_ex
        return loss

    def learn(
            self,
            total_iter: int,
            agent_learning_steps: Union[int, float],
            gradient_steps: int,
            max_agent_iter: int,
            agent_callback=None,
            callback=None,
            early_stop=True,
            **kwargs
    ):
        for itr in range(total_iter):
            self._reset_agent(**self.env_kwargs)
            with logger.accumulate_means(f"agent_{itr}"):
                for agent_steps in range(max_agent_iter):
                    loss_diff = self.cal_loss(**kwargs).item()
                    logger.record("loss_diff", loss_diff)
                    logger.record("agent_steps", agent_steps, exclude="tensorboard")
                    logger.dump(agent_steps)
                    if loss_diff > 0 and early_stop:
                        break
                    self.agent.learn(
                        total_timesteps=int(agent_learning_steps), reset_num_timesteps=False, callback=agent_callback)
                    logger.dump(step=self.agent.num_timesteps)
            losses = []
            with logger.accumulate_means(f"reward_{itr}"):
                self.reward_net.train()
                for rew_steps in range(gradient_steps):
                    loss = self.cal_loss(**kwargs)
                    losses.append(loss.item())
                    logger.record("steps", rew_steps, exclude="tensorboard")
                    logger.record("loss", loss.item())
                    logger.dump(rew_steps)
                    # TODO: Is there any smart way that breaks reward learning?
                    if loss.item() < -0.05 and early_stop:
                        break
                    self.reward_net.optimizer.zero_grad()
                    loss.backward()
                    self.reward_net.optimizer.step()
            if callback:
                callback(self, itr)


class GuidedCostLearning(MaxEntIRL):
    def transition_is(self, transition: Transitions) -> th.Tensor:
        if self.use_action_as_input:
            np_input = np.concatenate([transition.obs, transition.acts], axis=1)
            th_input = th.from_numpy(np_input).double()
        else:
            th_input = th.from_numpy(transition.obs).double()
        costs = -th.sum(self.reward_net(th_input))
        log_probs = th.sum(get_trajectories_probs(transition, self.agent.policy))
        return -costs + log_probs

    def cal_loss(self, **kwargs) -> th.Tensor:
        expert_trans = deepcopy(self.expert_transitions)
        IOCLoss1 = -self.mean_transition_reward(expert_trans)
        agent_trajs = self.rollout_from_agent(**kwargs)
        losses = th.zeros(len(agent_trajs))
        for i, traj in enumerate(agent_trajs):
            losses[i] = self.transition_is(flatten_trajectories([traj]))
        IOCLoss2 = th.logsumexp(losses, 0)
        return IOCLoss1 + IOCLoss2

    def learn(
            self,
            total_iter: int,
            agent_learning_steps: Union[int, float],
            gradient_steps: int,
            max_agent_iter: int,
            agent_callback=None,
            callback=None,
            early_stop=False,
            **kwargs
    ):
        super(GuidedCostLearning, self).learn(
            total_iter=total_iter,
            agent_learning_steps=agent_learning_steps,
            gradient_steps=gradient_steps,
            max_agent_iter=max_agent_iter,
            agent_callback=agent_callback,
            callback=callback,
            early_stop=early_stop,
            **kwargs,
        )
