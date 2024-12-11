"""
Slightly modified version of stable-baseline3 for my project
"""
import numpy as np
import torch as th
from typing import Tuple

from stable_baselines3.sac.policies import SACPolicy, CnnPolicy

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACPolicyCustom(SACPolicy):
    def get_log_prob_from_act(self, observation: np.ndarray, action: np.ndarray) -> th.Tensor:
        obs = th.from_numpy(observation).to(self.device)
        acts = th.from_numpy(action).to(self.device)
        mu, log_std, _ = self.actor.get_action_dist_params(obs)
        distribution = self.actor.action_dist.proba_distribution(mu, log_std)
        log_prob = distribution.log_prob(acts)
        return log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        mu, log_std, _ = self.actor.get_action_dist_params(obs)
        distribution = self.actor.action_dist.proba_distribution(mu, log_std)
        log_prob = distribution.log_prob(actions)
        return _, log_prob, -log_prob.mean()


MlpPolicy = SACPolicyCustom
CnnPolicy = CnnPolicy
