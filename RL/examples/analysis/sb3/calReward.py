import numpy as np
from matplotlib import pyplot as plt

from common.path_config import LOG_DIR
from common.sb3.util import load_result
from common.analyzer import exec_policy


def draw_bar_graph(xs, values):
    fig = plt.figure(figsize=[6.4, 4.8*len(values)])
    for i in range(len(values)):
        fig.add_subplot(len(values), 1, i+1)

    for value, ax in zip(values, fig.axes):
        ind = np.arange(value.shape[0])
        width = 0.6 / len(xs)
        for xi, x in enumerate(xs):
            i = xi - (len(xs) - 1) / 2
            ax.bar(ind + width*i, value[:, xi], width, label=x)
        # ax.set_ylim([0, 50])
        ax.set_xticks(ind)
        ax.set_xticklabels(['3', '6', '9', '12', '15'])
        ax.legend()
    fig.tight_layout()
    fig.show()


def reward_fn(obs, tqs, config):
    ank_max = config['env_kwargs']['ankle_torque_max']
    tqcr = config['env_kwargs']['tqcost_ratio']
    tqr = config['env_kwargs']['tq_ratio']
    ar = config['env_kwargs']['ank_ratio']
    velr = config['env_kwargs']['vel_ratio']
    limL = 10 ** (config['env_kwargs']['limLevel'] * ((-5) - (-2)) + (-2))
    # limL = 10 ** -4.5
    rews = []
    for ob, tq, comx in zip(obs, tqs, config['comx']):
        ankr = ar * ((ob[:360, 0] ** 2).sum())
        dankr = ar * velr*(ob[:360, 2] ** 2).sum()
        hipr = (1-ar) * (ob[:360, 1] ** 2).sum()
        dhipr = (1-ar) * velr * (ob[:360, 3] ** 2).sum()
        act = tq / ank_max
        tar = tqcr * tqr * (act[:, 0] ** 2).sum()
        thr = tqcr * (1-tqr) * (act[:, 1] ** 2).sum()
        postq = np.maximum(tq[:, 0] / ank_max, 0.)
        negtq = np.minimum(tq[:, 0] / ank_max, 0.)
        tqlr = tqcr * (limL * (1 / ((postq - 1) ** 2 + limL) + 1 / ((negtq + 0.5) ** 2 + limL))).sum()
        comr = limL * (1 / (comx - 0.15) ** 2 + limL).sum()
        rews.append([ankr+hipr, dankr+dhipr, 10*comr, 100*tqlr])

    return np.array(rews)


if __name__ == "__main__":
    load_kwargs = {
        "env_type": "IDP",
        "env_id": "MinEffort",
        "algo_type": "ppo",
        "trials": [1, 11, 21, 26, 33],
        "policy_num": 5,
        "tmp_num": 80,
        "log_dir": LOG_DIR / "erectCost" / "constNotEarly0823",
        "name_tail": "_MinEffort_ptb1to7/delay_passive/limLevel_50/const1.0_vel0.1_atm100",
    }

    env, agent, loaded_results = load_result(**load_kwargs)
    obs1, _, _, _, ifs = exec_policy(env, agent, render=None, deterministic=True, repeat_num=len(load_kwargs['trials']), infos=['torque', 'comx'])
    tqs1 = ifs['torque']
    comx1 = ifs['comx']
    config = loaded_results['config']
    config['comx'] = comx1
    rws1 = reward_fn(obs1, tqs1, config)
    load_kwargs = {
        "env_type": "IDP",
        "env_id": "MinEffort",
        "algo_type": "ppo",
        "trials": [1, 11, 21, 26, 33],
        "policy_num": 5,
        "tmp_num": 80,
        "log_dir": LOG_DIR / "erectCost" / "constNotEarly0823",
        "name_tail": "_MinEffort_ptb1to7/delay_passive/limLevel_50/const1.0_vel0.1_atm100",
    }

    env, agent, loaded_results = load_result(**load_kwargs)
    obs1, _, _, _, ifs = exec_policy(env, agent, render=None, deterministic=True, repeat_num=len(load_kwargs['trials']), infos=['torque', 'comx'])
    tqs1 = ifs['torque']
    comx1 = ifs['comx']
    config = loaded_results['config']
    config['comx'] = comx1
    rws2 = reward_fn(obs1, tqs1, config)
    # rws2 = reward_fn(loaded_results['states'], loaded_results['torques'], config)
    draw_bar_graph(["th", "dth", "comr", "tqlr"], [rws1, rws2])
