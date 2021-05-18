import os
import pickle
from algo.torch.ppo import PPO
from common.util import make_env
from common.verification import CostMap

if __name__ == "__main__":
    env_type = "IP"
    algo_type = "ppo"
    device = "cpu"
    name = "IP_custom"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = make_env(f"{name}-v2", use_vec_env=False, n_steps=600, sub="sub01")
    expt_env = make_env(f"{name}-v2", use_vec_env=False, n_steps=600, sub="sub01")
    ana_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name + "2")
    model_dir = os.path.join(ana_dir, "6", "model")
    with open(model_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f).double()

    def cost_fn(*args, **kwargs):
        reward = disc.reward_net.base_reward_net(*args, **kwargs)
        return -reward
    agent = PPO.load(model_dir + "/gen.zip")
    # expt = def_policy(env_type, expt_env)
    expt = PPO.load(f"../RL/{env_type}/tmp/log/{name}/ppo/policies_2/model.pkl")
    # expt = PPO.load(f"tmp/log/{env_type}/ppo/forward/model/extra_ppo0.zip")
    cost_map = CostMap(cost_fn, env, agent, expt_env, expt)
