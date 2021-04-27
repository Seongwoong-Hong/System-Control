import os
import numpy as np

from scipy import io
from matplotlib import pyplot as plt

from algo.torch.ppo import PPO
from common.util import make_env

if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    device = "cpu"
    current_path = os.path.dirname(__file__)
    sub = "sub01"
    ana_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_test")

    pltqs = []
    if env_type == "HPC":
        for i in [0, 5, 10, 15, 20, 25, 30]:
            file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]

    env = make_env(f"{env_type}_custom-v0", use_vec_env=False, n_steps=600, pltqs=pltqs)

    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    R = 1e-6 * np.array([[1, 0],
                         [0, 1]])

    num_list, rew_list = [], []
    num = 0
    while os.path.isfile(ana_dir + "/{}/model/gen.zip".format(num)):
        policy = PPO.load(ana_dir + "/{}/model/gen.zip".format(num))
        rew = 0
        for _ in range(7):
            obs = env.reset()
            done = False
            while not done:
                act, _ = policy.predict(obs, deterministic=True)
                rew += obs[:4] @ Q @ obs[:4] + act @ R @ act
                obs, _, done, _ = env.step(act)
        rew_list.append(7 / rew)
        num_list.append(num)
        num += 1
        env.close()
    pre = 0
    while pre < len(num_list):
        post = min(len(num_list), pre+25)
        plt.plot(num_list[pre:post], rew_list[pre:post])
        plt.show()
        pre = post
