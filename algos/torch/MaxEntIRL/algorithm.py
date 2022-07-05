from copy import deepcopy
from typing import List, Dict, Optional, Union, Tuple, Sequence

import gym
import numpy as np
import cvxpy as cp
import torch as th
from imitation.data.rollout import make_sample_until, flatten_trajectories
from imitation.data.types import Trajectory
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor

from algos.torch.MaxEntIRL import RewardNet, CNNRewardNet
from common.wrappers import RewardInputNormalizeWrapper
from common.rollouts import get_trajectories_probs, generate_trajectories_without_shuffle


class MaxEntIRL:
    def __init__(
            self,
            env,
            feature_fn,
            agent,
            expert_trajectories: List[Trajectory],
            device: str = 'cpu',
            eval_env=None,
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
        self.env_kwargs = env_kwargs
        self.rew_kwargs = rew_kwargs
        self.eval_env = eval_env
        self.agent_trajectories = []
        self.expert_trajectories = expert_trajectories

        self._setup()
        self.itr = 0

        if self.env_kwargs is None:
            self.env_kwargs = {}
        if self.eval_env is None:
            self.eval_env = env
        if self.rew_kwargs is None:
            self.rew_kwargs = {}

        inp = self.env.observation_space.sample()
        if self.use_action_as_input:
            inp = np.append(inp, self.env.action_space.sample())
        inp = feature_fn(th.from_numpy(inp).reshape(1, -1))

        reward_net_type = rew_kwargs.pop("type", None)
        if reward_net_type is None or reward_net_type is "ann":
            self.reward_net = RewardNet(
                inp=inp.shape[1],
                arch=rew_arch,
                feature_fn=feature_fn,
                use_action_as_inp=self.use_action_as_input,
                device=self.device,
                **self.rew_kwargs
            )
        elif reward_net_type is "cnn":
            self.reward_net = CNNRewardNet(
                inp=inp.shape[1],
                arch=rew_arch,
                feature_fn=feature_fn,
                use_action_as_inp=self.use_action_as_input,
                device=self.device,
                **self.rew_kwargs
            )
        else:
            raise NotImplementedError("Not implemented reward net type")

    def _setup(self):
        traj_len = len(self.expert_trajectories[0].acts)
        trans = flatten_trajectories(self.expert_trajectories)
        self.expert_reward_inp = trans.obs
        self.expert_gammas = th.Tensor([self.agent.gamma ** (i % traj_len) for i in range(len(trans))]).to(self.device)
        if self.use_action_as_input:
            self.expert_reward_inp = np.concatenate([self.expert_reward_inp, trans.acts], axis=1)

        self.init_D = th.zeros([self.agent.policy.obs_size], dtype=th.float32).to(self.device)
        init_obs, _ = self.eval_env.get_init_vector()
        init_idx = self.eval_env.get_idx_from_obs(init_obs)
        for i in range(len(init_idx)):
            self.init_D[init_idx[i]] = (init_idx == init_idx[i]).sum() / len(init_idx)

    def _reset_agent(self, **kwargs):
        reward_wrapper = kwargs.pop("reward_wrapper", RewardInputNormalizeWrapper)
        norm_wrapper = kwargs.pop("vec_normalizer", None)
        num_envs = kwargs.pop("num_envs", 1)
        self.wrap_env = reward_wrapper(self.env, self.reward_net.eval())
        self.wrap_eval_env = reward_wrapper(self.eval_env, self.reward_net.eval())
        if norm_wrapper:
            self.wrap_env = norm_wrapper(DummyVecEnv([lambda: Monitor(deepcopy(self.wrap_env))]), **kwargs)
        else:
            self.wrap_env = DummyVecEnv([lambda: Monitor(deepcopy(self.wrap_env)) for _ in range(num_envs)])
            self.vec_eval_env = DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(16)])
        self.agent.reset(self.wrap_env)

    def cal_expert_mean_reward(self) -> th.Tensor:
        """
        Calculate mean reward of expert transitions using current reward function
        :return: (transition mean rewards, None)
        """
        trans_rewards = self.expert_gammas * self.wrap_eval_env.reward(self.expert_reward_inp)
        return th.div(th.sum(trans_rewards), len(self.expert_trajectories))

    # noinspection PyPep8Naming
    def cal_agent_mean_reward(self) -> th.Tensor:
        D_prev = deepcopy(self.init_D)
        Dc = D_prev
        is_finite_agent = len(self.agent.policy.policy_table.shape) == 3
        if self.use_action_as_input:
            if is_finite_agent:
                Dc = Dc[None, :] * self.agent.policy.policy_table[0]
            else:
                Dc = Dc[None, :] * self.agent.policy.policy_table
        for t in range(1, self.env.spec.max_episode_steps):
            D = th.zeros_like(self.init_D).to(self.device)
            for a in range(self.agent.policy.act_size):
                if is_finite_agent:
                    D += self.agent.transition_mat[a] @ (D_prev * self.agent.policy.policy_table[t - 1, a])
                    # D += self.agent.transition_mat[a] @ (D_prev * self.agent.policy.policy_table[0, a])
                else:
                    D += self.agent.transition_mat[a] @ (D_prev * self.agent.policy.policy_table[a])
            if self.use_action_as_input:
                if is_finite_agent:
                    Dc += self.agent.policy.policy_table[t] * D[None, :] * self.agent.gamma ** t
                else:
                    Dc += self.agent.policy.policy_table * D[None, :] * self.agent.gamma ** t
            else:
                Dc += D * self.agent.gamma ** t
            D_prev = deepcopy(D)
        whole_reward_values = self.wrap_eval_env.get_reward_mat()
        return th.sum(Dc * whole_reward_values)

    def get_whole_states_from_env(self):
        """
        Get whole states from the 1D or 2D discrete environment
        :return: Set whole_state attribute
        """
        obs_space = self.env.observation_space
        if hasattr(self.env, "get_vectorized") and callable(getattr(self.env, "get_vectorized")):
            s_vec, _ = self.env.get_vectorized()
            self.whole_state = th.from_numpy(s_vec).float().to(self.device)
        elif isinstance(obs_space, gym.spaces.MultiDiscrete):
            x = np.meshgrid(*[range(nvec) for nvec in obs_space.nvec])
            # noinspection PyAttributeOutsideInit
            self.whole_state = th.FloatTensor([data.squeeze() for data in x]).t().to(self.device)
        else:
            raise NotImplementedError

    def train_reward_fn(self, max_gradient_steps, min_gradient_steps):
        losses = []
        for _ in range(max_gradient_steps):
            mean_expert_rewards = self.cal_expert_mean_reward()
            mean_agent_rewards = self.cal_agent_mean_reward()
            loss = mean_agent_rewards - mean_expert_rewards
            self.reward_net.optimizer.zero_grad()
            loss.backward()
            self.reward_net.optimizer.step()
            if self.reward_net.lr_scheduler and self.reward_net.lr_scheduler.get_last_lr()[0] > 1.5e-4:
                self.reward_net.lr_scheduler.step()
            logger.record("expert_reward", mean_expert_rewards.item())
            logger.record("agent_reward", mean_agent_rewards.item())
            logger.record("loss", loss.item())
            losses.append(loss.item())
        return np.mean(losses)

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
            reset_num_timesteps: bool = True,
            **kwargs
    ):
        if reset_num_timesteps or self.itr == 0:
            self.itr = 0
            self._reset_agent(**self.env_kwargs)
        else:
            total_iter += self.itr
        call_num = 0
        for param in self.reward_net.parameters():
            last_grad = th.zeros_like(param).view(-1)
            last_weight = th.zeros_like(param).view(-1)
        while self.itr < total_iter:
            with logger.accumulate_means(f"reward"):
                self.wrap_eval_env.train()
                mean_loss = self.train_reward_fn(max_gradient_steps, min_gradient_steps)
                weight_norm, grad_norm = 0.0, 0.0
                params_grad_data = []
                for param in self.reward_net.parameters():
                    weight_norm += param.norm().detach().item()
                    params_grad_data.append(param.grad.data.view(-1))
                    current_weight = param.data.view(-1)
                grad_norm = params_grad_data[-1].norm().item()
                grad_var_angle = th.arccos(th.clip(
                    (last_grad @ params_grad_data[-1]) / (last_grad.norm() * params_grad_data[-1].norm()),
                    min= -1 + 1e-8, max= +1 - 1e-8)
                ).item()
                weight_var_angle = th.arccos(th.clip(
                    (last_weight @ current_weight) / (last_weight.norm() * current_weight.norm()),
                    min= -1 + 1e-8, max= +1 - 1e-8)
                ).item()
                last_weight = deepcopy(current_weight)
                last_grad = deepcopy(params_grad_data[-1])
                logger.record("weight norm", weight_norm)
                logger.record("grad norm", grad_norm)
                logger.record("weight variant angle", weight_var_angle)
                logger.record("grad variant angle", grad_var_angle)
                logger.record("num iteration", self.itr, exclude="tensorboard")
                logger.dump(self.itr)
                if self.itr > 30 and np.abs(grad_norm) < 8e-4 and weight_var_angle < 5e-4 and early_stop:
                    break
            with logger.accumulate_means(f"agent"):
                self._reset_agent(**self.env_kwargs)
                for agent_steps in range(1, max_agent_iter + 1):
                    self.agent.learn(
                        total_timesteps=int(agent_learning_steps), reset_num_timesteps=False, callback=agent_callback)
                    logger.record("agent_steps", agent_steps, exclude="tensorboard")
                    logger.dump(self.itr)
            self.itr += 1
            if callback:
                callback(self, call_num)
                call_num += 1


class ContMaxEntIRL(MaxEntIRL):
    def _setup(self):
        self.expert_reward_inp = self.expert_trajectories[0].obs[:-1]
        if self.use_action_as_input:
            self.expert_reward_inp = np.append(self.expert_reward_inp, self.expert_trajectories[0].acts, axis=1)
        gammas = [self.agent.gamma ** i for i in range(len(self.expert_trajectories[0]))]
        for traj in self.expert_trajectories[1:]:
            inp = traj.obs[:-1]
            if self.use_action_as_input:
                inp = np.append(inp, traj.acts, axis=1)
            self.expert_reward_inp = np.append(self.expert_reward_inp, inp, axis=0)
            gammas += [self.agent.gamma ** i for i in range(len(traj))]
        self.expert_gammas = th.Tensor(gammas).to(self.device)

    # noinspection PyAttributeOutsideInit
    def collect_rollouts(self, n_episodes):
        """
        Collect trajectories using the agent
        :param n_episodes: Number of expert trajectories
        :return: Get new agent_trajectories attribute
        """
        print("Collecting rollouts from the current agent...")
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        self.agent_trajectories = generate_trajectories_without_shuffle(
            self.agent, self.vec_eval_env, sample_until, deterministic_policy=False)
        self.agent.set_env(self.wrap_env)

    def cal_agent_mean_reward(self) -> th.Tensor:
        rewards = th.zeros(len(self.agent_trajectories))
        for i, traj in enumerate(self.agent_trajectories):
            np_input = traj.obs[:-1]
            if self.use_action_as_input:
                acts = traj.acts
                if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                    acts = self.wrap_eval_env.action(acts)
                np_input = np.append(np_input, acts, axis=1)
            th_input = th.from_numpy(np_input).float().to(self.device)
            gammas = th.Tensor([self.agent.gamma ** i for i in range(len(traj))]).reshape(-1, 1).to(self.device)
            rewards[i] = (gammas * self.reward_net(th_input)).sum()
        return th.mean(rewards)

    def train_reward_fn(self, max_gradient_steps, min_gradient_steps):
        self.collect_rollouts(16)
        return super().train_reward_fn(max_gradient_steps, min_gradient_steps)


class GuidedCostLearning(ContMaxEntIRL):
    def cal_agent_mean_reward(self) -> th.Tensor:
        rewards, log_probs = th.zeros(len(self.agent_trajectories)), th.zeros(len(self.agent_trajectories))
        for i, traj in enumerate(self.agent_trajectories):
            np_input = traj.obs[:-1]
            if self.use_action_as_input:
                acts = traj.acts
                if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                    acts = self.wrap_eval_env.action(acts)
                np_input = np.append(np_input, acts, axis=1)
            th_input = th.from_numpy(np_input).float().to(self.device)
            gammas = th.Tensor([self.agent.gamma ** i for i in range(len(traj))]).reshape(-1, 1).to(self.device)
            rewards[i] = (gammas * self.reward_net(th_input)).sum()
            log_probs[i] = get_trajectories_probs(flatten_trajectories([traj]), self.agent.policy).sum()
        return th.logsumexp(rewards - log_probs, 0)


class APIRL(MaxEntIRL):
    def collect_rollouts(self, n_episodes):
        agent_mean_feature = deepcopy(self.agent_trajectories)
        self.agent_trajectories = []
        super().collect_rollouts(n_episodes)
        agent_mean_feature += self.traj_to_mean_feature(self.agent_trajectories, n_episodes)
        self.agent_trajectories = agent_mean_feature
        # noinspection PyAttributeOutsideInit
        self.current_agent_trajectories = deepcopy(self.agent_trajectories)

    def sample_and_cal_loss(self, n_episodes):
        expert_mean_feature = deepcopy(self.expert_trajectories)
        if isinstance(self.wrap_env, VecEnvWrapper):
            # noinspection PyAttributeOutsideInit
            self.vec_eval_env = deepcopy(self.wrap_env)
            self.vec_eval_env.set_venv(DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(n_episodes)]))
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes * self.expand_ratio)
        agent_trajectories = generate_trajectories_without_shuffle(
            self.agent, self.vec_eval_env, sample_until, deterministic_policy=True)
        self.agent.set_env(self.wrap_env)
        agent_mean_feature = self.traj_to_mean_feature(agent_trajectories, n_episodes)
        agent_rewards, _ = self.mean_transition_reward(agent_mean_feature)
        expert_rewards, _ = self.mean_transition_reward(expert_mean_feature)
        loss = th.mean(agent_rewards) - th.mean(expert_rewards)
        return loss

    def traj_to_mean_feature(self, traj: Sequence[Trajectory], n_episodes) -> List:
        """
        Calculate mean features of each agent or expert from trajectories
        :param traj: Collected trajectories
        :param n_episodes: The number of input collected trajectories for each agents
        :return: The mean feature of trajectories for each agents
        """
        mean_feature = []
        traj_len = len(traj[0].acts)
        gammas = np.array([self.agent.gamma ** (i % traj_len) for i in range(traj_len * len(traj))]).reshape(-1, 1)
        for i in range(0, len(traj), n_episodes):
            trans = flatten_trajectories(traj[i:i + n_episodes])
            # ft_array = np.zeros([traj_len * len(traj), self.agent.policy.obs_size])
            # ft_array[range(traj_len * len(traj)), self.agent.policy.obs_to_idx(trans.obs)] = 1
            ft_array = self.reward_net.feature_fn(trans.obs)
            mean_feature.append(np.sum(gammas * ft_array, axis=0) / len(traj))
            # mean_feature.append(np.sum(gammas * np.append(trans.obs, trans.obs ** 2, axis=1), axis=0) / len(traj))
        return mean_feature

    def mean_transition_reward(
            self,
            trajectories: Sequence[np.ndarray],
    ) -> Tuple[th.Tensor, None]:
        """
        Calculate mean of rewards of transitions from mean features of each agents
        :param trajectories: List of Trajectories
        :return: (agent mean reward, expert mean reward, None)
        """
        weight = self.reward_net.layers[0].weight.detach()
        rewards = th.dot(weight, th.from_numpy(np.array(trajectories)).float())
        return rewards, None

    # noinspection PyPep8Naming
    def train_reward_fn(self, max_gradient_steps, min_gradient_steps):
        t = cp.Variable()
        mu_e = np.array(deepcopy(self.expert_trajectories))
        mu = np.array(deepcopy(self.current_agent_trajectories))
        G = mu_e - mu
        w = cp.Variable(mu_e.shape[1])
        obj = cp.Maximize(t)
        constraints = [G @ w >= t, cp.quad_form(w, np.eye(mu_e.shape[1])) <= 1]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        with th.no_grad():
            self.reward_net.layers[0].weight = th.nn.Parameter(th.from_numpy(w.value).float(), requires_grad=True)
        logger.record("t", t.value.item())
        logger.dump()

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
            n_episodes: int = 10,
            **kwargs
    ):
        if isinstance(self.expert_trajectories[0], Trajectory):
            self.expert_trajectories = self.traj_to_mean_feature(self.expert_trajectories, n_episodes)
        self._reset_agent(**self.env_kwargs)
        for itr in range(total_iter):
            self.reward_net.reset_optim()
            with logger.accumulate_means(f"{itr}/reward"):
                self.collect_rollouts(n_episodes)
                self.train_reward_fn(max_gradient_steps, min_gradient_steps)
            with logger.accumulate_means(f"{itr}/agent"):
                self._reset_agent(**self.env_kwargs)
                reward_diffs = []
                for agent_steps in range(1, max_agent_iter + 1):
                    self.agent.learn(
                        total_timesteps=int(agent_learning_steps), reset_num_timesteps=False, callback=agent_callback)
                    logger.dump(step=self.agent.num_timesteps)
                    reward_diff = self.sample_and_cal_loss(n_episodes)
                    logger.record("loss_diff", reward_diff.item())
                    logger.record("agent_steps", agent_steps, exclude="tensorboard")
                    logger.dump(agent_steps)
                    reward_diffs.append(reward_diff.item())
                    if np.mean(reward_diffs[-min_agent_iter:]) >= 0 and \
                            reward_diffs[-1] >= 0.0 and \
                            agent_steps >= min_agent_iter and early_stop:
                        break
            if callback:
                callback(self, itr)
