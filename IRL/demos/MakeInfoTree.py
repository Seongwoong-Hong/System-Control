import os
import pickle
import numpy as np
import torch as th

from gym_envs.envs import DataBasedDiscretizationInfo
from gym_envs.envs.base.info_tree import *
from algos.torch.OptCont import LQRPolicy
from common.util import make_env
from common.wrappers import DiscretizeWrapper

class DiscIDPLQRPolicy(LQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = 0.878121, 1.047289
        l1 = 0.7970
        lc1, lc2 = 0.5084, 0.2814
        m1 ,m2 = 17.2955, 34.5085
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.Q = np.diag([1.0139, 0.1872182, 0.5639979, 0.1540204])
        self.R = np.diag([1.217065e-4, 0.917065e-4])
        self.gear = 1

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        action = super(DiscIDPLQRPolicy, self)._predict(observation, deterministic)
        high = th.from_numpy(self.env.action_space.high).float()
        low = th.from_numpy(self.env.action_space.low).float()
        action = th.clip(action, min=low, max=high)
        d_act = self.env.env_method("get_acts_from_idx", self.env.env_method("get_idx_from_acts", action.numpy())[0])[0]
        return th.from_numpy(d_act).float()

class IPLQRPolicy(LQRPolicy):
    def _build_env(self):
        g = 9.81
        m = 17.2955
        l = 0.7970
        lc = 0.5084
        I = 0.878121 + m * lc**2
        self.A, self.B = np.zeros([2, 2]), np.zeros([2, 1])
        self.A[0, 1] = 1
        self.A[1, 0] = m * g * lc / I
        self.B[1, 0] = 1 / I
        self.Q = np.diag([2.8139, 1.04872182])
        self.R = np.diag([1.617065e-4])
        self.gear = 1

    # def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    #     action = super(DiscIPLQRPolicy, self)._predict(observation, deterministic)
    #     high = th.from_numpy(self.env.action_space.high).float()
    #     low = th.from_numpy(self.env.action_space.low).float()
    #     action = th.clip(action, min=low, max=high)
    #     d_act = self.env.env_method("get_acts_from_idx", self.env.env_method("get_idx_from_acts", action.numpy())[0])[0]
    #     return th.from_numpy(d_act).float()


if __name__ == "__main__":
    demo_dir = os.path.abspath(".")
    env_type = "DiscretizedPendulum"
    name = f"{env_type}"
    obs_high = np.array([0.05, 0.3])
    obs_low = np.array([-0.05, -0.08])
    acts_high = np.array([40])
    acts_low = np.array([-30.])
    init_state = (obs_high + obs_low) / 2
    radius = (obs_high - obs_low) / 2
    obs_info_tree = InformationTree(init_state, radius, TwoDStateNode)
    init_act = (acts_high + acts_low) / 2
    act_rad = (acts_high - acts_low) / 2
    acts_info_tree = InformationTree(init_act, act_rad, OneDStateNode)
    obs_info = DataBasedDiscretizationInfo(obs_high, obs_low, obs_info_tree)
    acts_info = DataBasedDiscretizationInfo(acts_high, acts_low, acts_info_tree)
    env = make_env(f"{name}-v2", obs_info=obs_info, acts_info=acts_info)#, wrapper=DiscretizeWrapper)
    agent = IPLQRPolicy(env=env)

    for ep in range(200):
        agent.set_env(env)
        for _ in range(50):
            ob = env.reset()
            done = False
            while not done:
                obs_info_tree.count_visitation(ob)
                act, _ = agent.predict(ob)
                acts_info_tree.count_visitation(act)
                ob, _, done, _ = env.step(act)
        for _ in range(1):
            max_idx = np.argmax(obs_info_tree.visitation)
            obs_info_tree.divide_node(obs_info_tree.data[max_idx])
        if ep % 10 == 0:
            max_idx = np.argmax(acts_info_tree.visitation)
            acts_info_tree.divide_node(acts_info_tree.data[max_idx])
        env.obs_info.set_info(env.obs_high, env.obs_low, obs_info_tree)
        env.acts_info.set_info(env.max_torques, env.min_torques, acts_info_tree)
    dirname = f"{demo_dir}/{env_type}/databased_contlqr"
    os.makedirs(dirname, exist_ok=True)
    with open(f"{dirname}/obs_info_tree_200.pkl", "wb") as f:
        pickle.dump(obs_info_tree, f)
    with open(f"{dirname}/acts_info_tree_20.pkl", "wb") as f:
        pickle.dump(acts_info_tree, f)