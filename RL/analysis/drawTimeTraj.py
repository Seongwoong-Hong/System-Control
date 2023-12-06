import os
import numpy as np
from scipy import io
from matplotlib import pyplot as plt

from algos.torch.ppo import PPO
from common.util import make_env
from common.analyzer import exec_policy, video_record


if __name__ == "__main__":
    env_type = "IDP"
    env_id = f"{env_type}_MimicHuman"
    subj = "sub04"
    trials = range(26, 31)
    isPseudo = False
    use_norm = True
    policy_num = 3
    tmp_num = 15
    curri_order = None
    PDgain = np.array([500, 100])
    name_tail = f"_DeepMimic_ptb1to4/PD500100_ankLim"
    save_video = None
    except_trials = [13, 16]

    if isPseudo:
        env_type = "Pseudo" + env_type

    subpath = os.path.join("../..", "demos", env_type, subj, subj)
    states = [None for _ in range(35)]
    torques = [None for _ in range(35)]
    for trial in trials:
        humanData = io.loadmat(subpath + f"i{trial}.mat")
        bsp = humanData['bsp']
        states[trial - 1] = humanData['state']
        torques[trial - 1] = humanData['tq']
    for trial in except_trials:
        states[trial - 1] = None
        torques[trial - 1] = None

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
    t = np.linspace(0, 3, 361)
    fig = plt.figure(figsize=[4.4, 3.2])
    ax11 = fig.add_subplot(2, 2, 1)
    ax12 = fig.add_subplot(2, 2, 2)
    ax21 = fig.add_subplot(2, 2, 3)
    ax22 = fig.add_subplot(2, 2, 4)
    for idx, trial in enumerate(trials):
        if states[trial - 1] is not None:
            if trial <= 6:
                ax11.plot(t[:len(obs[idx][:-1])], obs[idx][:-1, 0], color=[0, 29/255, 120/255])
                ax12.plot(t[:len(states[trial - 1])], states[trial - 1][:, 0], color=[0, 29/255, 120/255])
                ax21.plot(t[:len(tqs[idx])], tqs[idx], color=[0, 29/255, 120/255])
                ax22.plot(t[:len(torques[trial - 1])], torques[trial - 1], color=[0, 29/255, 120/255])
            else:
                ax11.plot(t[:len(obs[idx][:-1])], obs[idx][:-1, 0], color=[60/255, 120/255, 210/255])
                ax12.plot(t[:len(states[trial - 1])], states[trial - 1][:, 0], color=[60/255, 120/255, 210/255])
                ax21.plot(t[:len(tqs[idx])], tqs[idx], color=[60/255, 120/255, 210/255])
                ax22.plot(t[:len(torques[trial - 1])], torques[trial - 1], color=[60/255, 120/255, 210/255])
    # ax11.set_ylim(ax12.get_ylim())
    ax21.set_ylim([-40, 110])
    ax22.set_ylim([-40, 110])
    for ax in [ax11, ax12, ax21, ax22]:
        ax.set_xlim([0, 3])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.show()

    if save_video is not None:
        video_record(imgs, f"videos/{save_video}.mp4", env.get_attr("dt")[0])
