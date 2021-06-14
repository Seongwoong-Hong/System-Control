"""
This file is here just to define MlpPolicy/CnnPolicy that work for PPO
Slightly modified version of stable-baseline3 for my project
"""
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy

from typing import Optional

import torch as th

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)


class ActorCriticPolicyCustom(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        log_std_range = kwargs.pop('log_std_range', None)
        if log_std_range is None:
            self.log_std_low, self.log_std_high = -5, 5
        else:
            self.log_std_low, self.log_std_high = log_std_range
        super(ActorCriticPolicyCustom, self).__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        log_std = th.clamp(self.log_std, self.log_std_low, self.log_std_high)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    def get_log_prob_from_act(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        log_prob = distribution.log_prob(action)
        return log_prob


MlpPolicy = ActorCriticPolicyCustom
CnnPolicy = ActorCriticCnnPolicy
