import os
import pickle
import torch as th
import numpy as np

from scipy import io

from IRL.scripts.project_policies import def_policy
from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import CostMap

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    device = "cpu"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sub = "sub01"
    name = "IDP_custom"
    write_txt = False

    pltqs = []
    test_len = 10
    if env_type == "HPC":
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            file = os.path.join(proj_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)

    agent_env = make_env(f"{name}-v0", use_vec_env=False, n_steps=600, pltqs=pltqs)
    expt_env = make_env(f"{name}-v0", use_vec_env=False, n_steps=600, pltqs=pltqs)
    # expt = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/ppo/policies_1/ppo0")
    expt = def_policy(env_type, expt_env)

    name = "no_lqr"
    ana_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name)
    model_dir = os.path.join(ana_dir, "model", "045")

    agent = SAC.load(model_dir + "/agent")

    def feature_fn(x):
        return x

    with open(model_dir + "/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f).double()

    def agent_cost_fn(*args):
        inp = th.cat([args[0], args[1]], dim=1)
        return -reward_fn(inp).item()

    def run_algo(env, algo):
        costs = []
        for _ in range(test_len):
            reward = 0
            obs = env.reset()
            done = False
            while not done:
                act, _ = algo.predict(obs, deterministic=False)
                obs, rew, done, _ = env.step(act)
                reward += rew
            costs.append(-reward)
        return costs

    agent_dict = {'env': agent_env, 'cost_fn': agent_cost_fn, 'algo': agent}
    expt_dict = {'env': expt_env, 'cost_fn': agent_cost_fn, 'algo': expt}
    agent_cost_for_agent, agent_cost_for_expt = CostMap.cal_cost(
        CostMap.process_agent(agent_dict) + CostMap.process_agent(expt_dict))
    expt_cost_for_agent = run_algo(agent_env, agent)
    expt_cost_for_expt = run_algo(expt_env, expt)
    print(f"agent_cost_for_agent mean: {np.mean(agent_cost_for_agent[1]):.3e}, std: {np.std(agent_cost_for_agent[1]):.3e}")
    print(f"agent_cost_for_expt mean: {np.mean(agent_cost_for_expt[1]):.3e}, std: {np.std(agent_cost_for_expt[1]):.3e}")
    print(f"expt_cost_for_agent mean: {np.mean(expt_cost_for_agent):.3e}, std: {np.std(expt_cost_for_agent):.3e}")
    print(f"expt_cost_for_expt mean: {np.mean(expt_cost_for_expt):.3e}, std: {np.std(expt_cost_for_expt):.3e}")
    if write_txt:
        f = open("MaxEntIRL_result.txt", "a")
        f.write(f"env: {name}\n")
        f.write(f"agent_cost_for_agent mean: {np.mean(agent_cost_for_agent[1]):.3e}, std: {np.std(agent_cost_for_agent[1]):.3e}\n")
        f.write(f"agent_cost_for_expt mean: {np.mean(agent_cost_for_expt[1]):.3e}, std: {np.std(agent_cost_for_expt[1]):.3e}\n")
        f.write(f"expt_cost_for_agent mean: {np.mean(expt_cost_for_agent):.3e}, std: {np.std(expt_cost_for_agent):.3e}\n")
        f.write(f"expt_cost_for_expt mean: {np.mean(expt_cost_for_expt):.3e}, std: {np.std(expt_cost_for_expt):.3e}\n\n")
        f.close()
