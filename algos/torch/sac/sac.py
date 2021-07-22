"""
Slightly modified version of stable-baseline3 for my project
"""
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC


class SACCustom(SAC):
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        super(SACCustom, self).__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                # add entropy term
                target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                q_backup = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            # using action from the replay buffer
            current_q_estimates = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        logger.record("train/mean_rewards", replay_data.rewards.mean().detach().item())
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def reset(self, env):
        self.init_kwargs['env'] = env
        super(SACCustom, self).__init__(*self.init_args, **self.init_kwargs)

    def reset_except_policy_param(self, env):
        self.init_kwargs['env'] = env
        param_vec = self.policy.parameters_to_vector()
        super(SACCustom, self).__init__(*self.init_args, **self.init_kwargs)
        self.policy.load_from_vector(param_vec)

    def set_env_and_reset_ent(self, env):
        self.num_timesteps = 0
        self.ent_coef = self.init_kwargs.pop('ent_coef', 'auto')
        self.set_env(env)
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def reset_std(self, env):
        self.init_kwargs['env'] = env
        state_dict = self.policy.state_dict()
        super(SACCustom, self).__init__(*self.init_args, **self.init_kwargs)
        state_dict['actor.log_std.weight'] = th.randn(state_dict['actor.log_std.weight'].size())
        state_dict['actor.log_std.bias'] = th.randn(state_dict['actor.log_std.bias'].size())
        self.policy.load_state_dict(state_dict)
