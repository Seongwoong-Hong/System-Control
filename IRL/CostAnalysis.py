import os
import pickle
from algo.torch.ppo import PPO
from IRL.project_policies import def_policy
from common.util import make_env
from common.verification import CostMap

if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    device = "cpu"
    current_path = os.path.dirname(__file__)
    env = make_env(f"{env_type}_custom-v0", use_vec_env=False, n_steps=600, sub="sub01")
    expt_env = make_env(f"{env_type}_custom-v0", use_vec_env=False, n_steps=600, sub="sub01")
    ana_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_test")
    model_dir = os.path.join(ana_dir, "1", "model")
    with open(model_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f).double()

    def cost_fn(*args, **kwargs):
        data = disc.reward_net.base_reward_net(*args, **kwargs)
        return -data
    agent = PPO.load(model_dir + "/gen.zip")
    expt = def_policy(env_type, expt_env)
    cost_map = CostMap(cost_fn, env, agent, expt_env, expt)
