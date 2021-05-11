import torch as th
from torch import nn
from copy import deepcopy
from typing import List, Dict, Optional

from algo.torch.sac import SAC, MlpPolicy
from imitation.data.rollout import make_sample_until, generate_trajectories, flatten_trajectories
from stable_baselines3.common.vec_env import DummyVecEnv


class RewardNet(nn.Module):
    def __init__(self,
                 lr: float,
                 arch: List[int],
                 optim_cls=th.optim.Adam,
                 activation_fn=th.nn.ReLU,
                 ):
        super(RewardNet, self).__init__()
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
        self.layers = nn.Sequential(*layers)
        self.optimizer = self.optim_cls(self.parameters(), lr)


class MaxEntIRL:
    def __init__(self,
                 env,
                 agent_learning_steps,
                 expert_transitions,
                 rew_kwargs: Dict,
                 sac_kwargs: Optional[Dict] = None,
                 ):
        self.env = env
        self.agent_learning_steps = agent_learning_steps
        if sac_kwargs is None:
            self.sac_kwargs = {}
        else:
            self.sac_kwargs = sac_kwargs
        self.expert_transitions = expert_transitions
        self.reward_net = RewardNet(**rew_kwargs)
        self._build_sac_agent(**self.sac_kwargs)

    def _build_sac_agent(self, **kwargs):
        self.agent = SAC(MlpPolicy,
                         env=self.env,
                         batch_size=256,
                         learning_starts=4096,
                         train_freq=2048,
                         n_episodes_rollout=-1,
                         gradient_steps=10,
                         gamma=0.99,
                         ent_coef='auto_0.05',
                         policy_kwargs={'net_arch': {'pi': [128, 128], 'qf': [128, 128]}},
                         **kwargs
                         )
        return self.agent

    def rollout_from_agent(self, n_episodes=10):
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        trajectories = generate_trajectories(
            self.agent, DummyVecEnv([lambda: self.env]), sample_until, deterministic_policy=False)
        return flatten_trajectories(trajectories)

    def cal_loss(self, agent_transitions):
        expert_transitions = deepcopy(self.expert_transitions)
        agent_ex = self.reward_net(agent_transitions)
        expert_ex = self.reward_net(expert_transitions)
        loss = agent_ex - expert_ex
        return loss

    def learn(self, total_iter: int = 1000, gradient_steps: int = 10):
        for itr in range(total_iter):
            self._build_sac_agent(**self.sac_kwargs)
            self.agent.learn(total_timesteps=self.agent_learning_steps)
            agent_samples = self.rollout_from_agent()
            for _ in range(gradient_steps):
                loss = self.cal_loss(agent_samples)
                self.reward_net.optimizer.zero_grad()
                loss.backward()
                self.reward_net.optimizer.step()


