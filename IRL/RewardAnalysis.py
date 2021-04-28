import os
import numpy as np

from scipy import io
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from algo.torch.ppo import PPO
from common.util import make_env

if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    device = "cpu"
    current_path = os.path.dirname(__file__)
    sub = "sub01"
    ana_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_test1")

    pltqs = []
    test_len = 5
    if env_type == "HPC":
        for i in [0, 5, 10, 15, 20, 25, 30]:
            file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)

    env = make_env(f"{env_type}_custom-v0", use_vec_env=False, n_steps=600, pltqs=pltqs)

    num_list, rew_list = [], []
    num = 0
    result_dict = {}

    def ana_fnc():
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        R = 1e-6 * np.array([[1, 0],
                             [0, 1]])
        param = 0
        for _ in range(test_len):
            obs = env.reset()
            done = False
            while not done:
                act, _ = policy.predict(obs, deterministic=True)
                param += obs[:4] @ Q @ obs[:4] + act @ R @ act
                obs, _, done, _ = env.step(act)
        return param
    while os.path.isfile(ana_dir + f"/{num}/model/gen.zip"):
        policy = PPO.load(ana_dir + f"/{num}/model/gen.zip")
        file = ana_dir + f"/{num}/model/hyper_parameters.txt"
        if not os.path.isfile(ana_dir + f"/{num}/model/result.txt"):
            cost = ana_fnc()
            f = open(ana_dir + f"/{num}/model/result.txt", "w")
            f.write(f"cost: {cost}\n")
            f.close()
        f = open(ana_dir + f"/{num}/model/result.txt", "r")
        saved_value = None
        for line in f.readlines():
            if "cost" in line:
                cost = float(line[line.find(":") + 1:-1])
        assert saved_value is None, "Can't find the cost value"
        rew_list.append(test_len / cost)
        num_list.append(num)
        num += 1
        f = open(file, "r")
        for line in f.readlines():
            param_name, param_value = line[:line.find(":")], float(line[line.find(":") + 1:-1])
            if param_name in result_dict:
                result_dict[param_name].append([param_value, test_len / cost])
            else:
                result_dict[param_name] = [[param_value, test_len / cost]]
        env.close()
    pre = 0
    while pre < len(num_list):
        post = min(len(num_list), pre+25)
        plt.plot(num_list[pre:post], rew_list[pre:post])
        plt.show()
        pre = post
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
