"""
Slightly modified version of stable-baseline3 for my project
"""
import torch as th

from stable_baselines3.sac.policies import SACPolicy, CnnPolicy

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACPolicyCustom(SACPolicy):
    def get_log_prob_from_act(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        mu, log_std, _ = self.actor.get_action_dist_params(observation)
        distribution = self.actor.action_dist.proba_distribution(mu, log_std)
        log_prob = distribution.log_prob(action)
        return log_prob


MlpPolicy = SACPolicyCustom
CnnPolicy = CnnPolicy
