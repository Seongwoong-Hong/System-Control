import gym_envs
import gym
import os

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO
from algo.torch.sac import SAC
from common.wrappers import CostWrapper
from common.verification import verify_policy, video_record
from scipy import io
from matplotlib import pyplot as plt


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    # name = "{}/{}/AIRL_hype_tune/104".format(env_type, algo_type)
    name = "{}/{}/2021-3-22-13-48-51".format(env_type, algo_type)
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    # costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
    # algo = PPO.load(model_dir + "/gen.zip")
    # algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
    algo = PPO.load(model_dir + "/extra_ppo1.zip")
    # algo = SAC.load(model_dir + "/sac{}.zip".format(num))
    sub = "sub01"
    expert_dir = os.path.join("..", "demos", env_type, sub + ".pkl")
    pltqs = []

    if env_type == "HPC":
        for i in [0, 5, 10, 15, 20, 25, 30]:
            file = os.path.join("..", "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=600, pltqs=pltqs)
    else:
        env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=600)

    ob_space = gym.spaces.Box(-10, 10, (4, ))
    exp = def_policy(env_type, env)

    for i in range(7):
        act_list, st_list, imgs = verify_policy(env, exp, 'human')
        pltq_list = st_list[:, 4:]
        obs_list = st_list[:, :4]
        plt.plot(obs_list[:, :2])
        plt.show()
        plt.plot(act_list)
        plt.show()
        plt.plot(pltq_list)
        plt.show()
    env.close()
