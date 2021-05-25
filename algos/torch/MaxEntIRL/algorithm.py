import torch as th
import numpy as np
from copy import deepcopy
from typing import List, Dict, Optional

from algos.torch.sac import SAC, MlpPolicy
from algos.torch.MaxEntIRL import RewardNet
from common.wrappers import RewardWrapper

from imitation.data.rollout import make_sample_until, generate_trajectories, flatten_trajectories
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv


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
        reward_wrapper = kwargs.get("reward_wrapper")
        if reward_wrapper:
            self.env = reward_wrapper(self.env, self.reward_net.eval())
            kwargs.pop('reward_wrapper')
        else:
            self.env = RewardWrapper(self.env, self.reward_net.eval())
        # TODO: Argument 들이 외부에서부터 입력되도록 변경. 파일의 형태로 넘겨주는 것 고려해 볼 것
        self.agent = SAC(
            MlpPolicy,
            env=self.env,
            batch_size=256,
            learning_starts=100,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            ent_coef='auto',
            device=self.device,
            policy_kwargs={'net_arch': {'pi': [32, 32], 'qf': [32, 32]}},
            **kwargs
        )
        return self.agent

    def rollout_from_agent(self, **kwargs):
        n_episodes = kwargs.get('n_episodes')
        if n_episodes is None:
            n_episodes = 10
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        trajectories = generate_trajectories(
            self.agent, DummyVecEnv([lambda: self.env]), sample_until, deterministic_policy=False)
        return flatten_trajectories(trajectories), len(flatten_trajectories(trajectories)) / n_episodes

    def mean_transition_reward(self, transition):
        if self.use_action_as_input:
            np_input = np.concatenate([transition.obs, transition.acts], axis=1)
            th_input = th.from_numpy(np_input)
        else:
            th_input = th.from_numpy(transition.obs)
        reward = self.reward_net(th_input)
        return reward.mean()

    def cal_loss(self, agent_transitions, total_timesteps):
        expert_transitions = deepcopy(self.expert_transitions)
        agent_ex = self.mean_transition_reward(agent_transitions)
        expert_ex = self.mean_transition_reward(expert_transitions)
        loss = agent_ex - expert_ex
        return loss * total_timesteps

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
        for itr in range(total_iter):
            self._build_sac_agent(**self.sac_kwargs)
            with logger.accumulate_means(f"agent_{itr}"):
                for agent_steps in range(max_sac_iter):
                    self.agent.learn(total_timesteps=self.agent_learning_steps, callback=agent_callback)
                    agent_samples, T = self.rollout_from_agent(**kwargs)
                    logger.record("loss_diff", self.cal_loss(agent_samples, T).item())
                    logger.record("agent_steps", agent_steps)
                    logger.dump(agent_steps)
                    if self.cal_loss(agent_samples, T) / T > 0:
                        break
            losses = []
            with logger.accumulate_means(f"reward_{itr}"):
                self.reward_net.train()
                for rew_steps in range(gradient_steps):
                    agent_samples, T = self.rollout_from_agent(**kwargs)
                    loss = self.cal_loss(agent_samples, T)
                    self.reward_net.optimizer.zero_grad()
                    loss.backward()
                    self.reward_net.optimizer.step()
                    losses.append(loss.item())
                    logger.record("steps", rew_steps + 1, exclude="tensorboard")
                    logger.record("loss", loss.item())
                    logger.dump(rew_steps)
                    if loss.item() < -50:
                        break
            if callback:
                callback(self, itr + 1)
            loss_logger.append(np.mean(losses))
        return loss_logger
