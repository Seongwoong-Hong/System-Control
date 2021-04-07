import gym_envs
import os
import numpy as np

from copy import deepcopy
from mujoco_py import GlfwContext
from scipy import io
from matplotlib import pyplot as plt

from algo.torch.ppo import PPO


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    env_id = "{}_custom-v1".format(env_type)
    n_steps = 600
    device = "cpu"
    current_path = os.path.dirname(__file__)
    GlfwContext(offscreen=True)
    sub = "sub01"
    expert_dir = os.path.join(current_path, "demos", env_type, sub + ".pkl")
    pltqs = []

    if env_type == "HPC":
        for i in range(35):
            file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        env = gym_envs.make("{}_custom-v1".format(env_type), n_steps=None, pltqs=pltqs)
    else:
        env = gym_envs.make("{}_custom-v1".format(env_type))

    ana_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_hype_tune")

    init_obs = env.reset().reshape(1, -1)
    init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
    init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
    init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
    init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)

    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    R = 1e-6 * np.array([[1, 0],
                         [0, 1]])

    num_list, rew_list = [], []
    num = 0
    while os.path.isdir(ana_dir + "/{}".format(num)):
        policy = PPO.load(ana_dir + "/{}/model/gen.zip".format(num))
        rew = 0
        for init_ob in init_obs:
            env.reset()
            env.set_state(init_ob[:2], init_ob[2:])
            obs = deepcopy(init_ob)
            for _ in range(600):
                act, _ = policy.predict(obs, deterministic=True)
                rew += obs @ Q @ obs + act @ R @ act
                obs, _, _, _ = env.step(act)
        rew_list.append(5/rew)
        num_list.append(num)
        num += 1

    plt.plot(num_list, rew_list)
    plt.show()
    env.close()
