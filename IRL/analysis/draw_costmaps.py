import os
import pickle

import torch as th
from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import CostMap

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    device = "cpu"
    name = "IDP_custom"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = make_env(f"{name}-v0", use_vec_env=False, n_steps=600, subpath="sub01")
    expt_env = make_env(f"{name}-v2", use_vec_env=False, n_steps=600, subpath="sub01")
    ana_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name + "_lqr")
    model_dir = os.path.join(ana_dir, "model")
    with open(model_dir + "/reward_net.pkl", "rb") as f:
        reward_net = pickle.load(f).double()

    def cost_fn(*args):
        inp = th.cat([args[0], args[1]], dim=1)
        return -reward_net(inp).item()
    agent = SAC.load(model_dir + "/agent.zip")
    # expt = def_policy(env_type, expt_env)
    expt = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/ppo/policies_1/ppo0")
    # expt = PPO.load(f"tmp/log/{env_type}/ppo/forward/model/extra_ppo0.zip")
    cost_map = CostMap(cost_fn, env, agent, expt_env, expt)
