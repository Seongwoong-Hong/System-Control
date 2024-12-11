import matplotlib.pyplot as plt
import numpy as np

from common.path_config import LOG_DIR
from common.sb3.util import load_result
from common.analyzer import exec_policy


def compare_hip_angle(trials, human_obs, sim_obs):
    max_hip_ang = []
    for hob, sob in zip(human_obs, sim_obs):
        max_hip_ang.append([np.max(-(hob[:, 1])), np.max(-(sob[:, 1]))])
    max_hip_ang = np.array(max_hip_ang)
    fig = plt.figure(figsize=[4.8, 6.4])
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(trials, max_hip_ang[:, 0])
    ax.scatter(trials, max_hip_ang[:, 1])
    fig.tight_layout()
    fig.show()


def draw_hip_angle_mean_std(obs):
    assert len(obs) == 35
    max_hip_ang = []
    for ob in obs:
        max_hip_ang.append(np.max(-ob[:, 1]))

    mean, std = [], []
    for perti in range(7):
        mean.append(np.mean(max_hip_ang[5*perti:5*(perti+1)]))
        std.append(np.std(max_hip_ang[5*perti:5*(perti+1)]))
    x = [3, 4.5, 6, 7.5, 9, 12, 15]
    fig, ax = plt.subplots()
    ax.errorbar(x, mean, yerr=std, fmt='o', color='b')
    ax.set_ylim([0, 0.35])
    ax.set_xticks([3, 6, 9, 12, 15])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=30)
    fig.tight_layout()
    fig.savefig('figure/limSatu', format='svg')


if __name__ == "__main__":
    load_kwargs = {
        "algo_type": "ppo",
        "env_type": "IDP",
        "env_id": "MinEffort",
        "trials": list(range(1, 36)),
        "policy_num": 10,
        "tmp_num": "opt",
        "log_dir": LOG_DIR / "learn_multi" / "check_in_morning",
        "name_tail": "_MinEffort_ptb1to7/limSatu",
    }
    save_video = None

    env, agent, loaded_result = load_result(**load_kwargs)

    render = "rgb_array"
    if save_video is None:
        render = None

    obs, _, _, _, _ = exec_policy(env, agent, render=None, deterministic=False, repeat_num=len(load_kwargs['trials']))
    human_obs = [state for state in loaded_result['states'] if state is not None]

    draw_hip_angle_mean_std(obs)
