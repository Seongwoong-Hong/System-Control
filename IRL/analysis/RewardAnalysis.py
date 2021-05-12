import os
import numpy as np

from scipy import io
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from algo.torch.ppo import PPO
from common.util import make_env, write_analyzed_result

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    device = "cpu"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sub = "sub01"
    name = "IDP_custom"
    ana_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name + "_easy_normal")

    pltqs = []
    test_len = 5
    if env_type == "HPC":
        for i in [0, 5, 10, 15, 20, 25, 30]:
            file = os.path.join(proj_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)

    env = make_env(f"{name}-v2", use_vec_env=False, n_steps=600, pltqs=pltqs)
    # expt_policy = PPO.load(f"tmp/log/{env_type}/{algo_type}/forward/model/extra_ppo0.zip")
    expt_policy = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/{algo_type}/policies_1/ppo0")

    def ana_fnc():
        param = 0
        for _ in range(test_len):
            obs = env.reset()
            done = False
            while not done:
                act, _ = policy.predict(obs, deterministic=True)
                obs, r, done, _ = env.step(act)
                param += r
        return {'cost': -param / test_len}

    num_list, rew_list = [], []
    num = 0
    result_dict = {}

    while os.path.isfile(ana_dir + f"/{num}/model/gen.zip"):
        policy = PPO.load(ana_dir + f"/{num}/model/gen.zip")
        file = ana_dir + f"/{num}/model/hyper_parameters.txt"
        write_analyzed_result(ana_fnc, ana_dir, iter_name=num)
        f = open(ana_dir + f"/{num}/model/result.txt", "r")
        cost = None
        for line in f.readlines():
            if "cost" in line:
                cost = float(line[line.find(":") + 1:-1])
        assert cost is not None, "Can't find the cost value"
        rew = -cost
        rew_list.append(rew)
        num_list.append(num)
        num += 1
        f = open(file, "r")
        for line in f.readlines():
            param_name, param_value = line[:line.find(":")], float(line[line.find(":") + 1:-1])
            if param_name in result_dict:
                result_dict[param_name].append([param_value, rew])
            else:
                result_dict[param_name] = [[param_value, rew]]
    policy = expt_policy
    expt_cost = ana_fnc()
    print(f"The reward of the expert is {-expt_cost['cost']}")
    env.close()
    for key, value in result_dict.items():
        fig = plt.figure()
        ax = fig.subplots()
        array = np.array(value)
        ax.set_title(key)
        ax.scatter(array[:, 0], array[:, 1])
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax.xaxis.set_major_locator(plt.AutoLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        ax.grid()
        plt.show()
    print(f"The best agent is {np.argmax(array[:, 1])} with a reward value {np.max(array[:, 1])}")
