import os
import pickle
import gym_envs
from algo.torch.ppo import PPO
from IRL.project_policies import def_policy
from common.verification import CostMap
from scipy import io

if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    env_id = "{}_custom-v0".format(env_type)
    n_steps = 600
    device = "cpu"
    current_path = os.path.dirname(__file__)
    sub = "sub01"
    expert_dir = os.path.join(current_path, "demos", env_type, sub + ".pkl")
    ana_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_div_test2")

    pltqs = []
    if env_type == "HPC":
        for i in range(35):
            file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        env = gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs)
    else:
        env = gym_envs.make(env_id, n_steps=n_steps)

    model_dir = os.path.join(ana_dir, "47", "model")
    with open(model_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f)
    cost_fn = disc.reward_net.base_reward_net.double()
    agent = PPO.load(model_dir + "/gen.zip")
    # agent = def_policy(env_type + "Div", env)
    cost_map = CostMap(env, cost_fn, agent)
