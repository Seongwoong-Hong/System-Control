import os
import pickle
import numpy as np

from scipy import io

from algo.torch.ppo import PPO
from common.util import make_env
from common.verification import CostMap

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    device = "cpu"
    saving = True
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sub = "sub01"
    name = "IDP_custom"

    pltqs = []
    test_len = 5
    if env_type == "HPC":
        for i in [0, 5, 10, 15, 20, 25, 30]:
            file = os.path.join(proj_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)

    agent_env = make_env(f"{name}-v2", use_vec_env=False, n_steps=600, pltqs=pltqs)
    expt_env = make_env(f"{name}-v2", use_vec_env=False, n_steps=600, sub="sub01")
    expt = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/{algo_type}/policies_1/ppo0")

    name += "_easy"
    ana_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name)
    model_dir = os.path.join(ana_dir, "49", "model")

    agent = PPO.load(model_dir + "/gen.zip")

    with open(model_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f).double()

    def agent_cost_fn(*args, **kwargs):
        reward = disc.reward_net.base_reward_net(*args, **kwargs)
        return -reward

    def run_algo(env, algo):
        costs = []
        for _ in range(10):
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
        CostMap.process_agent(agent_dict, n_episodes=10) + CostMap.process_agent(expt_dict, n_episodes=10))
    expt_cost_for_agent = run_algo(agent_env, agent)
    expt_cost_for_expt = run_algo(expt_env, expt)
    print(f"agent_cost_for_agent mean: {np.mean(agent_cost_for_agent[1]):.3e}, std: {np.std(agent_cost_for_agent[1]):.3e}")
    print(f"agent_cost_for_expt mean: {np.mean(agent_cost_for_expt[1]):.3e}, std: {np.std(agent_cost_for_expt[1]):.3e}")
    print(f"expt_cost_for_agent mean: {np.mean(expt_cost_for_agent):.3e}, std: {np.std(expt_cost_for_agent):.3e}")
    print(f"expt_cost_for_expt mean: {np.mean(expt_cost_for_expt):.3e}, std: {np.std(expt_cost_for_expt):.3e}")
    f = open("result_table.txt", "a")
    f.write(f"env: {name}\n")
    f.write(f"agent_cost_for_agent mean: {np.mean(agent_cost_for_agent[1]):.3e}, std: {np.std(agent_cost_for_agent[1]):.3e}\n")
    f.write(f"agent_cost_for_expt mean: {np.mean(agent_cost_for_expt[1]):.3e}, std: {np.std(agent_cost_for_expt[1]):.3e}\n")
    f.write(f"expt_cost_for_agent mean: {np.mean(expt_cost_for_agent):.3e}, std: {np.std(expt_cost_for_agent):.3e}\n")
    f.write(f"expt_cost_for_expt mean: {np.mean(expt_cost_for_expt):.3e}, std: {np.std(expt_cost_for_expt):.3e}\n\n")
    f.close()
