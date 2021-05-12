import torch as th
import numpy as np
from torch import nn
from copy import deepcopy
from typing import List, Dict, Optional

from algo.torch.sac import SAC, MlpPolicy
from common.wrappers import RewardWrapper

from imitation.data.rollout import make_sample_until, generate_trajectories, flatten_trajectories
from stable_baselines3.common.vec_env import DummyVecEnv


class RewardNet(nn.Module):
    def __init__(self,
                 lr: float,
                 arch: List[int],
                 device: str,
                 optim_cls=th.optim.Adam,
                 activation_fn=th.nn.ReLU,
                 ):
        super(RewardNet, self).__init__()
        self.device = device
        self.act_fnc = activation_fn
        self.optim_cls = optim_cls
        self._build(lr, arch)

    def _build(self, lr, arch):
        layers = []
        if self.act_fnc is not None:
            for i in range(len(arch) - 1):
                layers.append(nn.Linear(arch[i], arch[i + 1]))
                layers.append(self.act_fnc())
        else:
            for i in range(len(arch) - 1):
                layers.append(nn.Linear(arch[i], arch[i + 1]))
        layers.append(nn.Linear(arch[-1], 1, bias=False))
        self.layers = nn.Sequential(*layers).to(self.device)
        self.optimizer = self.optim_cls(self.parameters(), lr)

    def forward(self, x):
        return self.layers(x.to(self.device))


class MaxEntIRL:
    def __init__(self,
                 env,
                 agent_learning_steps,
                 expert_transitions,
                 device='cpu',
                 rew_lr: float = 1e-3,
                 rew_arch: List[int] = None,
                 use_action_as_input: bool = True,
                 rew_kwargs: Optional[Dict] = None,
                 sac_kwargs: Optional[Dict] = None,
                 ):
        self.env = env
        self.agent_learning_steps = agent_learning_steps
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
        self.reward_net = RewardNet(lr=rew_lr, arch=rew_arch, device=self.device, **self.rew_kwargs).double()

    def _build_sac_agent(self, **kwargs):
        self.env = RewardWrapper(self.env, self.reward_net)
        # TODO: Argument 들이 외부에서부터 입력되도록 변경. 파일의 형태로 넘겨주는 것 고려해 볼 것
        self.agent = SAC(MlpPolicy,
                         env=self.env,
                         batch_size=256,
                         learning_starts=100,
                         train_freq=1,
                         n_episodes_rollout=-1,
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
            n_episodes = len(self.expert_transitions)
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        trajectories = generate_trajectories(
            self.agent, DummyVecEnv([lambda: self.env]), sample_until, deterministic_policy=False)
        return flatten_trajectories(trajectories)

    def mean_transition_reward(self, transition):
        if self.use_action_as_input:
            np_input = np.concatenate([transition.obs, transition.acts], axis=1)
            th_input = th.from_numpy(np_input)
        else:
            th_input = th.from_numpy(transition.obs)
        reward = self.reward_net(th_input)
        return reward.mean()

    def cal_loss(self, agent_transitions):
        expert_transitions = deepcopy(self.expert_transitions)
        agent_ex = self.mean_transition_reward(agent_transitions)
        expert_ex = self.mean_transition_reward(expert_transitions)
        loss = agent_ex - expert_ex
        return loss

    def learn(self, total_iter: int = 1000, gradient_steps: int = 10, **kwargs):
        # TODO: Add loggers for the tensorboard and the command line
        # TODO: Make callbacks work
        loss_logger = []
        for itr in range(total_iter):
            self._build_sac_agent(**self.sac_kwargs)
            self.agent.learn(total_timesteps=self.agent_learning_steps)
            losses = []
            for _ in range(gradient_steps):
                agent_samples = self.rollout_from_agent(**kwargs)
                loss = self.cal_loss(agent_samples)
                losses.append(deepcopy(loss.item()))
                self.reward_net.optimizer.zero_grad()
                loss.backward()
                self.reward_net.optimizer.step()
            loss_logger.append(np.mean(losses))
        return loss_logger
