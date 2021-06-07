from copy import deepcopy
from typing import List, Dict, Optional

import numpy as np
import torch as th
from imitation.data.rollout import make_sample_until, generate_trajectories, flatten_trajectories
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algos.torch.MaxEntIRL import RewardNet
from algos.torch.sac import SAC, MlpPolicy
from common.wrappers import RewardWrapper


class MaxEntIRL:
    def __init__(
            self,
            env,
            agent_learning_steps_per_one_loop,
            expert_transitions,
            device='cpu',
            rew_lr: float = 1e-3,
            rew_arch: List[int] = None,
            use_action_as_input: bool = True,
            rew_kwargs: Optional[Dict] = None,
            sac_kwargs: Optional[Dict] = None,
    ):
        assert (
            logger.is_configured()
        ), "Requires call to imitation.util.logger.configure"
        self.env = env
        self.agent_learning_steps = agent_learning_steps_per_one_loop
        self.device = device
        self.use_action_as_input = use_action_as_input
        if sac_kwargs is None:
            self.sac_kwargs = {}
        else:
            self.sac_kwargs = sac_kwargs
        if rew_kwargs is None:
            self.rew_kwargs = {}
        else:
            self.rew_kwargs = rew_kwargs
        self.expert_transitions = expert_transitions
        inp = self.env.observation_space.shape[0]
        if self.use_action_as_input:
            inp += self.env.action_space.shape[0]
        self.reward_net = RewardNet(inp=inp, arch=rew_arch, lr=rew_lr, device=self.device, **self.rew_kwargs).double()

    def _build_sac_agent(self, **kwargs):
        reward_wrapper = kwargs.pop("reward_wrapper", RewardWrapper)
        self.reward_net.eval()
        self.wrap_env = reward_wrapper(self.env, self.reward_net)
        self.venv = VecNormalize(DummyVecEnv([lambda: self.wrap_env]))
        # TODO: Argument 들이 외부에서부터 입력되도록 변경. 파일의 형태로 넘겨주는 것 고려해 볼 것
        self.agent = SAC(
            MlpPolicy,
            env=self.venv,
            batch_size=256,
            learning_starts=100,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            ent_coef=0.1,
            device=self.device,
            policy_kwargs={'net_arch': {'pi': [32, 32], 'qf': [32, 32]}},
            **kwargs
        )
        return self.agent

    def rollout_from_agent(self, **kwargs):
        n_episodes = kwargs.pop('n_episodes', 10)
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        trajectories = generate_trajectories(
            self.agent, DummyVecEnv([lambda: self.wrap_env]), sample_until, deterministic_policy=False)
        return flatten_trajectories(trajectories)

    def mean_transition_reward(self, transition):
        if self.use_action_as_input:
            np_input = np.concatenate([transition.obs, transition.acts], axis=1)
            th_input = th.from_numpy(np_input)
        else:
            th_input = th.from_numpy(transition.obs)
        reward = self.reward_net(th_input)
        return reward.mean()

    def cal_loss(self, **kwargs):
        expert_transitions = deepcopy(self.expert_transitions)
        agent_transitions = self.rollout_from_agent(**kwargs)
        agent_ex = self.mean_transition_reward(agent_transitions)
        expert_ex = self.mean_transition_reward(expert_transitions)
        loss = agent_ex - expert_ex
        return loss

    def learn(
            self,
            total_iter: int = 10,
            gradient_steps: int = 100,
            max_sac_iter: int = 10,
            agent_callback=None,
            callback=None,
            **kwargs
    ):
        loss_logger = []
        self._build_sac_agent(**self.sac_kwargs)
        for itr in range(total_iter):
            with logger.accumulate_means(f"agent_{itr}"):
                for agent_steps in range(max_sac_iter):
                    self.agent.learn(total_timesteps=self.agent_learning_steps, callback=agent_callback)
                    logger.record("loss_diff", self.cal_loss(**kwargs).item())
                    logger.record("agent_steps", agent_steps)
                    logger.dump(agent_steps)
                    if self.cal_loss(**kwargs) > 0:
                        break
            losses = []
            with logger.accumulate_means(f"reward_{itr}"):
                self.reward_net.train()
                for rew_steps in range(gradient_steps):
                    loss = self.cal_loss(**kwargs)
                    losses.append(loss.item())
                    logger.record("steps", rew_steps + 1, exclude="tensorboard")
                    logger.record("loss", loss.item())
                    logger.dump(rew_steps)
                    # TODO: Is there any smart way that breaks reward learning?
                    if np.mean(losses[-5:]) < -0.1:
                        break
                    self.reward_net.optimizer.zero_grad()
                    loss.backward()
                    self.reward_net.optimizer.step()
            if callback:
                callback(self, itr + 1)
            loss_logger.append(np.mean(losses))
        return loss_logger
