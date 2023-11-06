import torch as th
import numpy as np
from scipy import linalg
from typing import Optional, Tuple
from torch.distributions import MultivariateNormal
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from algos.torch.MaxEntIRL.reward_net import *


class LQRPolicy(BasePolicy):
    def __init__(
            self,
            env,
            gamma = 1.0,
            alpha = 1.0,
            noise_lv: float = 0.1,
            observation_space=None,
            action_space=None,
            device='cpu',
            **kwargs,
    ):
        if observation_space is None:
            observation_space = env.observation_space
        if action_space is None:
            action_space = env.action_space
        super(LQRPolicy, self).__init__(observation_space, action_space)

        self.env = None
        self.gamma = gamma
        self.alpha = alpha
        self.set_env(env)
        self.noise_lv = noise_lv
        self._build_env()
        self._get_gains()

    def set_env(self, env):
        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: env])
        else:
            self.env = env

    def reset(self, env):
        self.set_env(env)
        self._build_env()
        self._get_gains()

    def _get_gains(self):
        X = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        if self.R.shape[0] >= 2:
            K = (np.linalg.inv(self.R) @ (self.B.T @ X))
        else:
            K = (1 / self.R * (self.B.T @ X)).reshape(-1)
        self.K = th.from_numpy(K).double().to(self.device)

    def get_vec_normalize_env(self):
        return None

    def _build_env(self):
        # define self.A, self.B, self.Q, self.R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def forward(self):
        return None

    def learn(self, *args, **kwargs):
        reward_net = self.env.envs[0].rwfn
        if isinstance(self.env.envs[0].rwfn, LURewardNet):
            w_th = reward_net.w_th.detach().cpu().numpy()
            w_tq = reward_net.w_tq.detach().cpu().numpy()
            qu, ru = np.zeros([4, 4]), np.zeros([2, 2])
            qu[0, :] = w_th[:4]
            qu[1, 1:] = w_th[4:7]
            qu[2, 2:] = w_th[7:9]
            qu[3, 3] = w_th[9]
            ru[0, :] = w_tq[:2]
            ru[1, 1] = w_tq[2]
            self.Q = qu.T @ qu
            self.R = ru.T @ ru
        elif isinstance(self.env.envs[0].rwfn, XXRewardNet):
            # weights = np.square(reward_net.reward_layer.weight.cpu().detach().numpy().flatten())
            weights = reward_net.reward_layer.weight.cpu().detach().numpy().flatten()
            self.Q = np.diag(weights[:self.observation_space.shape[0]])
            self.R1 = np.diag(weights[-2*self.action_space.shape[0]:-self.action_space.shape[0]])
            self.R2 = np.diag(weights[-self.action_space.shape[0]:])
        else:
            if isinstance(self.env.envs[0].rwfn, QuadraticRewardNet):
                # weights = np.square(reward_net.reward_layer.weight.cpu().detach().numpy().flatten())
                weights = reward_net.reward_layer.weight.cpu().detach().numpy().flatten()
            else:
                weights = reward_net.layers[0].weight.cpu().detach().numpy().flatten()
            self.Q = np.diag(weights[:self.observation_space.shape[0]])
            self.R = np.diag(weights[self.observation_space.shape[0]:]) / (self.gear ** 2)
        self._get_gains()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        noise = 0
        if not deterministic:
            noise = self.noise_lv * np.random.randn(*self.K.shape)

        return -1 / self.gear * ((self.K + noise) @ observation.T).reshape(1, -1)


class DiscreteLQRPolicy(LQRPolicy):
    def _build_env(self):
        # define self.A, self.B, self.Q, self.R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def _get_gains(self):
        X = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        if self.R.shape[0] >= 2:
            K = -(np.linalg.inv(self.B.T @ X @ self.B + self.R) @ (self.B.T @ X @ self.A))
        else:
            K = -(1 / (self.B.T @ X @ self.B + self.R) * (self.B.T @ X @ self.A))
        self.K = th.from_numpy(K).float().to(self.device)
        self.COV = th.from_numpy(np.linalg.inv((self.B.T @ X @ self.B + self.R) / self.alpha)).float()

    def get_log_prob_from_act(self, trans_obs, trans_acts):
        mean_acts = th.from_numpy(trans_obs).float() @ self.K.T
        distributions = MultivariateNormal(mean_acts, self.COV)
        log_probs = distributions.log_prob(th.from_numpy(trans_acts * self.gear).float())
        return log_probs

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        act = observation.float() @ self.K.T
        if not deterministic:
            eps = th.randn([observation.shape[0], self.env.action_space.shape[0]])
            if self.COV.shape[0] == 1:
                act += eps @ th.sqrt(self.COV)
            else:
                act += eps @ th.linalg.cholesky(self.COV).T
        act = th.clamp(act, min=-self.gear, max=self.gear)
        return act / self.gear


class FiniteLQRPolicy(LQRPolicy):
    def _build_env(self):
        # define self.A, self.B, self.Q, self.R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def _get_gains(self):
        self.vvs = np.zeros_like(self.Q)[None, ...].repeat(self.max_t + 1, axis=0)
        self.vs = np.zeros(self.Q.shape[0])[None, ...].repeat(self.max_t + 1, axis=0)
        self.kks = np.zeros_like(self.B.T)[None, ...].repeat(self.max_t, axis=0)
        self.ks = np.zeros(self.R.shape[0])[None, ...].repeat(self.max_t, axis=0)
        self.cov_series = np.zeros_like(self.R)[None, ...].repeat(self.max_t, axis=0)
        self.vvs[-1] = self.Q
        self.vs[-1] = self.q[-1]
        for t in reversed(range(self.max_t)):
            inv_mat = np.linalg.inv(self.R + self.B.T @ self.vvs[t + 1] @ self.B)
            self.cov_series[t] = inv_mat / self.alpha
            self.kks[t] = -inv_mat @ self.B.T @ self.vvs[t + 1] @ self.A
            self.ks[t] = -(self.r[t] + self.vs[t + 1] @ self.B) @ inv_mat
            self.vs[t] = (self.q[t] + self.vs[t + 1] @ self.A -
                          (self.r[t] + self.vs[t + 1] @ self.B) @ inv_mat @ self.B.T @ self.vvs[t + 1] @ self.A)
            self.vvs[t] = (self.Q + self.A.T @ self.vvs[t + 1] @ self.A -
                           self.A.T @ self.vvs[t + 1] @ self.B @ inv_mat @ self.B.T @ self.vvs[t + 1] @ self.A)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        obs_list, act_list, rew_list = [], [], []
        self.env.reset()
        self.env.env_method("set_state", observation.squeeze())
        obs_list.append(observation.squeeze())
        done = False
        t = 0
        while not done:
            act = observation @ self.kks[t].T + self.ks[t]
            if not deterministic:
                eps = np.random.standard_normal(self.env.action_space.shape)
                if np.any(np.linalg.eigvals(self.cov_series[t]) <= 0):
                    print(self.cov_series[t])
                act += eps @ np.linalg.cholesky(self.cov_series[t]).T
                act = np.clip(act, a_min=-self.gear, a_max=self.gear)
            observation, reward, done, info = self.env.step([act / self.gear])
            if done:
                observation = info[0]['terminal_observation']
            observation = observation.flatten()
            obs_list.append(observation)
            rew_list.append(reward.flatten())
            act_list.append(act.flatten())
            t += 1
        return np.array(obs_list), np.array(act_list) / self.gear, np.array(rew_list)


class DiffLQRPolicy(FiniteLQRPolicy):
    def _build_env(self):
        # define self.A, self.B, self.Q, self.R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def _get_gains(self):
        self.vvs = np.zeros_like(self.Q)[None, ...].repeat(self.max_t + 1, axis=0)
        self.wws = np.zeros_like(self.B.T)[None, ...].repeat(self.max_t + 1, axis=0)
        self.uus = np.zeros_like(self.R1)[None, ...].repeat(self.max_t + 1, axis=0)
        self.kks = np.zeros_like(self.B.T)[None, ...].repeat(self.max_t, axis=0)
        self.ks = np.zeros_like(self.R1)[None, ...].repeat(self.max_t, axis=0)
        # self.cov_series = np.zeros_like(self.R)[None, ...].repeat(self.max_t, axis=0)
        self.vvs[-1] = self.Q
        self.wws[-1] = np.zeros_like(self.B.T)
        self.uus[-1] = self.R2
        for t in reversed(range(self.max_t)):
            inv_mat = np.linalg.inv(self.R1 + self.R2 + self.B.T @ self.vvs[t + 1] @ self.B + 2 * self.wws[t + 1] @ self.B + self.uus[t + 1])
            # self.cov_series[t] = inv_mat / self.alpha
            self.kks[t] = -inv_mat @ (self.B.T @ self.vvs[t + 1] @ self.A + self.wws[t + 1] @ self.A)
            self.ks[t] = inv_mat @ self.R2
            self.vvs[t] = (self.Q + self.A.T @ self.vvs[t + 1] @ self.A -
                           ((self.A.T @ self.vvs[t + 1] @ self.B + self.A.T @ self.wws[t + 1].T) @
                            inv_mat @ (self.B.T @ self.vvs[t + 1] @ self.A + self.wws[t + 1] @ self.A)))
            self.wws[t] = self.R2 @ inv_mat @ (self.B.T @ self.vvs[t + 1] @ self.A + self.wws[t + 1] @ self.A)
            self.uus[t] = self.R2 - self.R2 @ inv_mat @ self.R2

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        obs_list, act_list, rew_list = [], [], []
        self.env.reset()
        self.env.env_method("set_state", observation.squeeze())
        act = observation @ self.kks[0].T
        if not deterministic:
            eps = np.random.standard_normal(self.env.action_space.shape)
            if np.any(np.linalg.eigvals(self.cov_series[0]) <= 0):
                print(self.cov_series[0])
            act += eps @ np.linalg.cholesky(self.cov_series[0]).T
            act = np.clip(act, a_min=-self.gear, a_max=self.gear)
        t = 1
        while True:
            obs_list.append(observation.squeeze())
            observation, reward, done, info = self.env.step([act / self.gear])
            act_list.append(act)
            rew_list.append(reward.flatten())
            if done:
                observation = info[0]['terminal_observation']
                obs_list.append(observation.flatten())
                break
            act = observation @ self.kks[0].T + act @ self.ks[0].T
            if not deterministic:
                eps = np.random.standard_normal(self.env.action_space.shape)
                if np.any(np.linalg.eigvals(self.cov_series[t]) <= 0):
                    print(self.cov_series[t])
                act += eps @ np.linalg.cholesky(self.cov_series[t]).T
                act = np.clip(act, a_min=-self.gear, a_max=self.gear)
            act = act.flatten()
            t += 1
        return np.array(obs_list), np.array(act_list) / self.gear, np.array(rew_list)


class iterLQRPolicy(LQRPolicy):
    def _build_env(self):
        # f_x, u_x : dynamics 미분
        # l_x, l_u, l_xx, l_uu, l_ux, lf_x, lf_xx: cost 미분
        # 위 값들을 정의해야 함
        # 또한 v_x 및 v_xx의 initialization 필요
        raise NotImplementedError

    def _get_gains(self):
        ob_size, act_size = self.observation_space.shape[0], self.action_space.shape[0]
        self.cov_series = th.zeros([act_size, act_size])[None, ...].repeat(self.max_t, 1, 1)
        self.v_x = th.zeros(ob_size)[None, ...].repeat(self.max_t + 1, 1)
        self.v_xx = th.zeros([ob_size, ob_size])[None, ...].repeat(self.max_t + 1, 1, 1)
        observation = self.env.reset()

        obs_list = np.append(observation.reshape(1, -1), np.zeros([self.max_t, self.observation_space.shape[0]]), axis=0)
        acts_list = np.zeros([self.max_t, self.action_space.shape[0]])

        epoch = 0
        while True:
            ks, kks = self.backward_pass(obs_list, acts_list)
            prev_obs_list = obs_list.copy()
            obs_list, acts_list = self.forward_pass(obs_list, acts_list, ks, kks)
            if np.linalg.norm(prev_obs_list - obs_list) < 0.1 or epoch >= 20:
                break
            epoch += 1
        self.ks, self.kks = th.stack(ks), th.stack(kks)

    def learn(self, *args, **kwargs):
        reward_net = self.env.envs[0].rwfn
        assert isinstance(reward_net, QuadraticRewardNet) or isinstance(reward_net, CNNRewardNet)
        feature_layer = reward_net.feature_layers.eval()
        weights = reward_net.reward_layer.weight.detach().flatten().square()
        self.Q = weights[:-reward_net.len_act_w]
        self.R = weights[-reward_net.len_act_w:]
        self.l = lambda x, u: (th.sum(feature_layer(reward_net.feature_fn(x)[None, None, :]).flatten() * self.Q, dim=-1) +
                               th.sum(reward_net.feature_fn(u / self.gear) * self.R, dim=-1))
        self.lf = lambda x: th.sum(feature_layer(reward_net.feature_fn(x)[None, None, :]).flatten() * self.Q, dim=-1)
        self._get_gains()

    def backward_pass(self, obs, acts):
        x = th.nn.Parameter(th.from_numpy(obs[-1]).float())
        lf = self.lf(x)
        dlf_x = th.autograd.grad(lf, x, create_graph=True)[0]
        ddlf_xx = th.stack([th.autograd.grad(dlf_x[i], x, retain_graph=True)[0] for i in range(len(dlf_x))])
        self.v_x[-1] = dlf_x
        self.v_xx[-1] = ddlf_xx
        ks, kks = [], []
        for t in reversed(range(self.max_t)):
            x, u = th.nn.Parameter(th.from_numpy(obs[t]).float()), th.nn.Parameter(th.from_numpy(acts[t]).float())
            l = self.l(x, u)
            dl_x = th.autograd.grad(l, x, create_graph=True)[0]
            ddl_xx = th.stack([th.autograd.grad(dl_x[i], x, retain_graph=True)[0] for i in range(len(dl_x))])
            dl_u = th.autograd.grad(l, u, create_graph=True)[0]
            ddl_uu = th.stack([th.autograd.grad(dl_u[i], u, retain_graph=True)[0] for i in range(len(dl_u))])
            f_x_t = self.f_x(obs[t], acts[t])
            f_u_t = self.f_u(obs[t], acts[t])
            q_x = dl_x + f_x_t.T @ self.v_x[t + 1]
            q_u = dl_u + f_u_t.T @ self.v_x[t + 1]
            q_xx = ddl_xx + (f_x_t.T @ self.v_xx[t + 1]) @ f_x_t
            q_uu = ddl_uu + (f_u_t.T @ self.v_xx[t + 1]) @ f_u_t
            q_ux = (f_u_t.T @ self.v_xx[t + 1]) @ f_x_t
            q_xu = (f_x_t.T @ self.v_xx[t + 1]) @ f_u_t
            self.cov_series[t] = th.linalg.inv(q_uu / self.alpha)
            inv_q_uu = th.linalg.inv(q_uu)
            k = -inv_q_uu @ q_u
            kk = -inv_q_uu @ q_ux
            self.v_x[t] = (q_x + q_xu @ k).detach()
            self.v_xx[t] = (q_xx + q_xu @ kk).detach()
            ks.append(k.detach())
            kks.append(kk.detach())
        ks.reverse()
        kks.reverse()
        return ks, kks

    def forward_pass(self, obs, acts, ks, kks):
        obs_high = self.env.envs[0].high
        obs_low = self.env.envs[0].low
        obs_hat = np.array(obs).copy()
        acts_hat = np.array(acts).copy()
        for t in range(self.max_t):
            k, kk = ks[t].numpy(), kks[t].numpy()
            act_hat = np.clip(acts[t] + k + np.matmul(kk, (obs_hat[t] - obs[t])), -self.gear, self.gear)
            acts_hat[t] = act_hat.flatten()
            obs_hat[t + 1] = obs_hat[t] @ self.A.T + acts_hat[t] @ self.B.T
        obs_hat = np.clip(obs_hat, obs_low, obs_high)
        return obs_hat, acts_hat

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        act = self.ks[0] + observation.float() @ self.kks[0].T
        if not deterministic:
            eps = th.randn([observation.shape[0], self.env.action_space.shape[0]])
            act += eps @ th.linalg.cholesky(self.cov_series[0]).T
        act = th.clamp(act, min=-self.gear, max=self.gear)
        return act / self.gear
