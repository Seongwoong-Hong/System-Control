import os

import numpy as np
from scipy import io
from matplotlib import pyplot as plt
from common.util import make_env
from algos.torch.ppo import PPO
from common.analyzer import exec_policy, video_record


if __name__ == "__main__":
    env_type = "IP"
    env_id = f"{env_type}_custom"
    subj = "sub04"
    trials = [5, 10, 15, 20, 25, 30, 35]
    isPseudo = True
    use_norm = True
    policy_num = 2
    tmp_num = 15
    PDgain = np.array([1000, 200])
    name_tail = f"_DeepMimic_actionSkip_ptb1to6/PD{PDgain[0]}{PDgain[1]}_ankLim"
    save_video = None

    if isPseudo:
        env_type = "Pseudo" + env_type
    subpath = os.path.join("../..", "demos", env_type, subj, subj)
    states = [None for _ in range(35)]
    for trial in trials:
        humanData = io.loadmat(subpath + f"i{trial}.mat")
        bsp = humanData['bsp']
        states[trial - 1] = humanData['state']
    if use_norm:
        env_type += "_norm"
        model_dir = os.path.join("..", "scripts", "tmp", "log", f"{env_type}", "ppo" + name_tail, f"policies_{policy_num}")
        norm_pkl_path = model_dir + f"/normalization_{tmp_num}.pkl"
    else:
        model_dir = os.path.join("..", "scripts", "tmp", "log", f"{env_type}", "ppo" + name_tail, f"policies_{policy_num}")
        norm_pkl_path = False

    env = make_env(f"{env_id}-v0", bsp=bsp, humanStates=states, use_norm=norm_pkl_path, PDgain=PDgain)

    agent = PPO.load(model_dir + f"/agent_{tmp_num}")

    render = "rgb_array"
    if save_video is None:
        render = None
    obs, acts, _, imgs = exec_policy(env, agent, render=render, deterministic=False, repeat_num=10)
    if use_norm:
        norm_obs = []
        for ob in obs:
            norm_obs.append(env.unnormalize_obs(ob))
        del obs
        obs = norm_obs

    for idx, trial in enumerate(trials):
        plt.plot(obs[idx][:-1, 0], 'b')
        plt.plot(states[trial - 1][:, 0], 'k')
    plt.show()
    for idx, trial in enumerate(trials):
        plt.plot(acts[idx], 'b')
    plt.show()

    if save_video is not None:
        video_record(imgs, f"videos/{save_video}.mp4", env.get_attr("dt")[0])
