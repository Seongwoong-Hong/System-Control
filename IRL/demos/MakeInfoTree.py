import os
import pickle
import numpy as np
import torch as th

from gym_envs.envs import InformationTree, TwoDStateNode, OneDStateNode
from algos.torch.OptCont import LQRPolicy
from common.util import make_env
from common.wrappers import DiscretizeWrapper


class DiscIPLQRPolicy(LQRPolicy):
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

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        action = super(DiscIPLQRPolicy, self)._predict(observation, deterministic)
        high = th.from_numpy(self.env.action_space.high).float()
        low = th.from_numpy(self.env.action_space.low).float()
        action = th.clip(action, min=low, max=high)
        d_act = self.env.env_method("get_acts_from_idx", self.env.env_method("get_idx_from_acts", action.numpy())[0])[0]
        return th.from_numpy(d_act).float()


if __name__ == "__main__":
    demo_dir = os.path.abspath(".")
    env_type = "DiscretizedPendulum"
    name = f"{env_type}_DataBased"
    env = make_env(f"{name}-v2", N=[29, 29], NT=[21], wrapper=DiscretizeWrapper)
    agent = DiscIPLQRPolicy(env=env)
    init_state = (env.obs_high + env.obs_low) / 2
    radius = (env.obs_high - env.obs_low) / 2
    obs_info_tree = InformationTree(init_state, radius, TwoDStateNode)
    init_act = (env.max_torques + env.min_torques) / 2
    act_rad = (env.max_torques - env.min_torques) / 2
    acts_info_tree = InformationTree(init_act, act_rad, OneDStateNode)
    for ep in range(300):
        env.obs_info.set_info(env.obs_high, env.obs_low, obs_info_tree)
        env.acts_info.set_info(env.max_torques, env.min_torques, acts_info_tree)
        agent.set_env(env)
        for _ in range(5):
            ob = env.reset()
            done = False
            while not done:
                obs_info_tree.find_target_node(ob)
                act, _ = agent.predict(ob)
                acts_info_tree.find_target_node(act)
                ob, _, done, _ = env.step(act)
        for _ in range(5):
            max_idx = np.argmax(obs_info_tree.visitation)
            obs_info_tree.divide_node(obs_info_tree.data[max_idx].value)
        if ep % 6 == 0:
            max_idx = np.argmax(acts_info_tree.visitation)
            acts_info_tree.divide_node(acts_info_tree.data[max_idx].value)
    with open(f"{demo_dir}/{env_type}/databased_lqr/obs_info_tree_1500.pkl", "wb") as f:
        pickle.dump(obs_info_tree, f)
    with open(f"{demo_dir}/{env_type}/databased_lqr/acts_info_tree_50.pkl", "wb") as f:
        pickle.dump(acts_info_tree, f)