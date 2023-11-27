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
    trials = range(1, 21)
    isPseudo = False
    use_norm = True
    policy_num = 1
    tmp_num = 14
    curri_order = None
    PDgain = np.array([1000, 200])
    name_tail = f"_DeepMimic_actionSkip_ptb1to4/PD{PDgain[0]}{PDgain[1]}_ankLim"
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
    if curri_order is not None:
        model_dir += f"/curriculum_{curri_order}"

    if use_norm:
        norm_pkl_path = model_dir + f"/normalization_{tmp_num}.pkl"
    else:
        norm_pkl_path = False

    env = make_env(f"{env_id}-v0", bsp=bsp, humanStates=states, use_norm=norm_pkl_path, PDgain=PDgain)

    agent = PPO.load(model_dir + f"/agent_{tmp_num}")

    render = "rgb_array"
    if save_video is None:
        render = None
    obs, acts, _, imgs, tqs = exec_policy(env, agent, render=render, deterministic=True, repeat_num=len(trials))
    if use_norm:
        norm_obs = []
        for ob in obs:
            norm_obs.append(env.unnormalize_obs(ob))
        del obs
        obs = norm_obs
    fig = plt.figure(figsize=[6.4, 9.6])
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    for idx, trial in enumerate(trials):
        ax1.plot(obs[idx][:-1, 0], 'b')
        ax1.plot(states[trial - 1][:, 0], 'k')
        ax2.plot(acts[idx][:], 'b')
        # ax2.plot(acts[idx][:, 1], 'k')
        ax3.plot(tqs[idx][:], 'b')
        # ax3.plot(tqs[idx][:, 1], 'k')
    plt.show()

    if save_video is not None:
        video_record(imgs, f"videos/{save_video}.mp4", env.get_attr("dt")[0])
