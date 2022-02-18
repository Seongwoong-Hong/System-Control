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
            **kwargs,
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
        self.expert_trajectories = expert_trajectories
        self.expert_transitions = flatten_trajectories(expert_trajectories)

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

        self.agent_trajectories = []
        self.expand_ratio = 20
        self.itr = 0

        if self.env_kwargs is None:
            self.env_kwargs = {}
        if self.eval_env is None:
            self.eval_env = env
        if self.rew_kwargs is None:
            self.rew_kwargs = {}

        self.env.reset()
        _, _, _, info = self.env.step(self.env.action_space.sample())
        inp = info['obs']
        if self.use_action_as_input:
            inp = np.append(inp, self.env.action_space.sample())
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
            )
        elif RNet_type is "cnn":
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
            self.vec_eval_env = DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(self.expand_ratio)])
        self.agent.reset(self.wrap_env)

    def collect_rollouts(self, n_episodes):
        """
        Collect trajectories using the agent
        :param n_episodes: Number of expert trajectories
        :return: Get new agent_trajectories attribute
        """
        print("Collecting rollouts from the current agent...")
        if isinstance(self.wrap_env, VecEnvWrapper):
            self.vec_eval_env = deepcopy(self.wrap_env)
            self.vec_eval_env.set_venv(
                DummyVecEnv([lambda: deepcopy(self.wrap_eval_env) for _ in range(self.expand_ratio)]))
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes * self.expand_ratio)
        self.agent_trajectories = generate_trajectories_without_shuffle(
            self.agent, self.vec_eval_env, sample_until, deterministic_policy=False)
        self.agent.set_env(self.wrap_env)

    def cal_expert_mean_reward(self) -> th.Tensor:
        """
        Calculate mean reward of expert transitions using current reward function
        :return: (transition mean rewards, None)
        """
        trans_rewards = self.expert_gammas * self.wrap_eval_env.reward(self.expert_reward_inp)
        return th.sum(trans_rewards) / len(self.expert_trajectories)

    def cal_state_visitation(self) -> th.Tensor:
        D_prev = deepcopy(self.init_D)
        Dc = D_prev
        if self.use_action_as_input:
            Dc = Dc[None, :] * self.agent.policy.policy_table[0]
        for t in range(1, self.env.spec.max_episode_steps):
            D = th.zeros_like(self.init_D).to(self.device)
            for a in range(self.agent.policy.act_size):
                D += self.agent.transition_mat[a] @ (D_prev * self.agent.policy.policy_table[t - 1, a])
            if self.use_action_as_input:
                Dc += self.agent.policy.policy_table[t] * D[None, :] * self.agent.gamma ** t
            else:
                Dc += D * self.agent.gamma ** t
            D_prev = deepcopy(D)
        return Dc

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
            self.whole_state = th.FloatTensor([data.squeeze() for data in x]).t().to(self.device)
        else:
            raise NotImplementedError

    def train_reward_fn(self, max_gradient_steps, min_gradient_steps):
        mean_expert_rewards = self.cal_expert_mean_reward()
        Dc = self.cal_state_visitation()
        whole_reward_values = self.wrap_eval_env.get_reward_mat()
        loss = th.sum(Dc * whole_reward_values) - mean_expert_rewards
        self.reward_net.optimizer.zero_grad()
        loss.backward()
        self.reward_net.optimizer.step()
        logger.record("expert_reward", mean_expert_rewards.item())
        logger.record("agent_reward", th.sum(Dc * whole_reward_values).item())
        logger.record("loss", loss.item())
        return loss.item()

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
            n_episodes: int = 10,
            reset_num_timesteps: bool = True,
            **kwargs
    ):
        if reset_num_timesteps or self.itr == 0:
            self.get_whole_states_from_env()
            self.itr = 0
            self._reset_agent(**self.env_kwargs)
        else:
            total_iter += self.itr
        call_num = 0
        while self.itr < total_iter:
            with logger.accumulate_means(f"reward"):
                self.wrap_eval_env.train()
                mean_loss = self.train_reward_fn(max_gradient_steps, min_gradient_steps)
                weight_norm, grad_norm = 0.0, 0.0
                for param in self.reward_net.parameters():
                    weight_norm += param.norm().detach().item()
                    grad_norm += param.grad.norm().item()
                logger.record("weight norm", weight_norm)
                logger.record("grad norm", grad_norm)
                logger.record("num iteration", self.itr, exclude="tensorboard")
                logger.dump(self.itr)
                if np.abs(mean_loss) < 1e-2 and self.itr > 30 and np.abs(grad_norm) < 1e-3 and early_stop:
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


class GuidedCostLearning(MaxEntIRL):
    def collect_rollouts(self, n_agent_episodes):
        # self.agent_trajectories = []
        super().collect_rollouts(n_agent_episodes)

    def transition_is(self, trajectories: Sequence[Trajectory]) -> Tuple[th.Tensor, th.Tensor]:
        traj_len = len(trajectories[0].acts)
        transitions = flatten_trajectories(trajectories)
        if self.use_action_as_input:
            acts = transitions.acts
            if hasattr(self.wrap_eval_env, "action") and callable(self.wrap_eval_env.action):
                acts = self.wrap_eval_env.action(acts)
            np_input = np.concatenate([transitions.obs, acts], axis=1)
            th_input = th.from_numpy(np_input).float()
        else:
            th_input = th.from_numpy(transitions.obs).float()
        gammas = th.FloatTensor([self.agent.gamma ** (i % traj_len) for i in range(len(th_input))]).to(self.device)
        trans_reward = gammas * self.reward_net(th_input).flatten()
        trans_log_prob = get_trajectories_probs(transitions, self.agent.policy)
        rewards, log_probs = th.zeros(len(trajectories)), th.zeros(len(trajectories))
        for i in range(len(trajectories)):
            rewards[i] = trans_reward[i*traj_len:(i+1)*traj_len].sum()
            log_probs[i] = trans_log_prob[i*traj_len:(i+1)*traj_len].sum()
        return rewards, log_probs

    def cal_loss(self, n_episodes) -> Tuple:
        expert_trajectories = deepcopy(self.expert_trajectories)
        demo_trajs = expert_trajectories + deepcopy(self.current_agent_trajectories)
        rewards, log_probs = self.transition_is(demo_trajs)
        expt_rewards, _ = self.mean_transition_reward(expert_trajectories)
        loss = th.logsumexp(rewards - log_probs, 0) - expt_rewards.mean()
        logger.record("log_probs_mean", th.mean(log_probs).item())
        logger.record("log_probs_var", th.var(log_probs).item())
        weight_norm = 0.0
        for param in self.reward_net.parameters():
            weight_norm += param.norm().detach().item()
        return loss, weight_norm, None

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
        super(GuidedCostLearning, self).learn(
            total_iter=total_iter,
            agent_learning_steps=agent_learning_steps,
            max_gradient_steps=max_gradient_steps,
            min_gradient_steps=min_gradient_steps,
            max_agent_iter=max_agent_iter,
            min_agent_iter=min_agent_iter,
            agent_callback=agent_callback,
            callback=callback,
            early_stop=early_stop,
            n_episodes=n_episodes,
            **kwargs,
        )


class APIRL(MaxEntIRL):
    def collect_rollouts(self, n_episodes):
        agent_mean_feature = deepcopy(self.agent_trajectories)
        self.agent_trajectories = []
        super().collect_rollouts(n_episodes)
        agent_mean_feature += self.traj_to_mean_feature(self.agent_trajectories, n_episodes)
        self.agent_trajectories = agent_mean_feature
        self.current_agent_trajectories = deepcopy(self.agent_trajectories)

    def sample_and_cal_loss(self, n_episodes):
        expert_mean_feature = deepcopy(self.expert_trajectories)
        if isinstance(self.wrap_env, VecEnvWrapper):
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
