from matplotlib import pyplot as plt

from common.path_config import LOG_DIR
from common.sb3.util import load_result
from common.analyzer import exec_policy, video_record
from RL.policies import *

def draw_time_trajs(obs, acts, x=None):
    if x is None:
        x = np.arange(obs.shape[1])
    rT, cT = 4, max(round(obs.shape[-1] / 2), acts.shape[-1])
    fig = plt.figure(figsize=[4.0 * cT, 10.0])
    for i in range(rT * cT):
        fig.add_subplot(rT, cT, i+1)
    for i in range(obs.shape[-1]):
        fig.axes[i].plot(x, np.rad2deg(obs[:, :-1, i].T))
    for i in range(acts.shape[-1]):
        fig.axes[obs.shape[-1]+i].plot(x, acts[:, :, i].T)

    fig.tight_layout()
    fig.show()
    return fig

def draw_time_trajs_with_humandata(trials, obs, tqs, states, torques, showfig=True, title=None):
    t = np.linspace(0, 5, 601)
    cs = [[60/255, 120/255, 210/255], [30/255, 75/255, 155/255], [0., 30/255, 120/255], [0., 0., 45/255]]
    fig = plt.figure(figsize=[4.4, 8.4])
    for i in range(12):
        fig.add_subplot(6, 2, i+1)
    for idx, trial in enumerate(trials):
        cidx = (trial - 1)//10
        for stidx in range(4):
            fig.axes[2*stidx].plot(t[:len(obs[idx][:-1])], np.rad2deg(obs[idx][:-1, stidx]), color=cs[cidx])
            fig.axes[2*stidx+1].plot(t[:len(states[idx])], np.rad2deg(states[idx][:, stidx]), color=cs[cidx])
        for actidx in range(2):
            fig.axes[2*actidx+8].plot(t[:len(tqs[idx])], tqs[idx][:, actidx], color=cs[cidx])
            fig.axes[2*actidx+9].plot(t[:len(torques[idx])], torques[idx][:, actidx], color=cs[cidx])
    fig.axes[0].set_ylim(fig.axes[1].get_ylim())
    # fig.axes[0].set_ylim([-15, 5])
    fig.axes[2].set_ylim([fig.axes[3].get_ylim()[0]*1.2, fig.axes[3].get_ylim()[1]*2])
    fig.axes[3].set_ylim([fig.axes[3].get_ylim()[0]*1.2, fig.axes[3].get_ylim()[1]*2])
    fig.axes[4].set_ylim(fig.axes[5].get_ylim())
    fig.axes[6].set_ylim(fig.axes[7].get_ylim())
    fig.axes[8].set_ylim(fig.axes[9].get_ylim()[0], fig.axes[9].get_ylim()[1])
    fig.axes[9].set_ylim(fig.axes[8].get_ylim()[0], fig.axes[8].get_ylim()[1])
    fig.axes[10].set_ylim([fig.axes[11].get_ylim()[0]*2, fig.axes[11].get_ylim()[1]])
    fig.axes[11].set_ylim([fig.axes[11].get_ylim()[0]*2, fig.axes[11].get_ylim()[1]])
    for ax in fig.axes:
        ax.set_xlim([0, 3])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    labels = [r"$\theta_{ank}$", r"$\theta_{hip}$", r"$\omega_{ank}$", r"$\omega_{hip}$", r"$T_{ank}$", r"$T_{hip}$"]
    for label, i in zip(labels, range(0, len(fig.axes), 2)):
        fig.axes[i].set_ylabel(label, fontsize=15)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    if showfig:
        fig.show()
    return fig


def draw_versus_graph(trials, d1s, d2s):
    cs = [[60/255, 120/255, 210/255], [30/255, 75/255, 155/255], [0., 30/255, 120/255], [0., 0., 45/255]]
    fig = plt.figure(figsize=[3.2*len(d1s), 3.2])
    for i in range(len(d1s)):
        ax = fig.add_subplot(1, len(d1s), i+1)
        for idx, trial in enumerate(trials):
            cidx = (trial - 1) // 10
            ax.plot(d1s[i][idx], d2s[i][idx], color=cs[cidx])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.show()
    return fig


def draw_passive_torque(trials, tqs, ptqs, title):
    t = np.linspace(0, 5, 601)
    cs = [[60/255, 120/255, 210/255], [30/255, 75/255, 155/255], [0., 30/255, 120/255], [0., 0., 45/255]]
    fig = plt.figure(figsize=[6, 4])
    for i in range(6):
        fig.add_subplot(2, 3, i + 1)
    for idx, trial in enumerate(trials):
        cidx = (trial - 1) // 10
        for i in range(2):
            fig.axes[3*i].plot(t[:len(tqs[idx])], tqs[idx][:, i], color=cs[cidx])
            fig.axes[3*i+1].plot(t[:len(ptqs[idx])], ptqs[idx][:, i], color=cs[cidx])
            fig.axes[3*i+2].plot(t[:len(ptqs[idx])], tqs[idx][:, i] - ptqs[idx][:, i], color=cs[cidx])
            fig.axes[3*i+1].set_ylim(fig.axes[3*i].get_ylim())
            fig.axes[3*i+2].set_ylim(fig.axes[3*i].get_ylim())
            fig.axes[3*i].spines['right'].set_visible(False)
            fig.axes[3*i+1].spines['right'].set_visible(False)
            fig.axes[3*i+2].spines['right'].set_visible(False)
            fig.axes[3*i].spines['top'].set_visible(False)
            fig.axes[3*i+1].spines['top'].set_visible(False)
            fig.axes[3*i+2].spines['top'].set_visible(False)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    fig.show()
    return fig


if __name__ == "__main__":
    name_tail = "ppo_MinEffort_ptb1to7/delay_passive/limLevel_50/const1.0_vel0.1_atm100"
    model_dir = LOG_DIR / "erectCost" / "constNotEarly0823" / "IDP" / name_tail / "policies_3"

    # model_dir = LOG_DIR / "Cartpole_29-07-13-54"

    load_kwargs = {
        "env_type": "IDP",
        "env_id": "MinEffort",
        "log_dir": model_dir,
        "algo_num": 80,

        "env_kwargs": {
            "trials": [1, 2, 11, 12, 21, 22, 26, 27, 31, 33],
            "ankle_torque_max": 100,
            'ptb_act_time': 0.275,
            "stiffness": [300, 50],
            "damping": [30, 20],
            "delay": True,
            "delayed_time": 0.1,
            "ankle_limit": "soft"
        },
    }
    save_video = None

    env, agent, loaded_result = load_result(**load_kwargs)
    # env.envs[0].env.env.env.delay = False
    # env.set_attr("delay", False)
    # Linear Feedback Controller를 사용할 때는 normalization 확인하기

    # agent = LinearFeedbackPolicy(env, gain=np.array([[580.4426, 59.0801, 66.9362, 98.6479], [128.4063, 119.9887, 9.5562, 28.1239]]))
    # agent = LinearFeedbackPolicy(env, gain=np.array([[256.9201, 283.4496, 110.5109, 60.0833], [-22.1334, 188.7776, 30.5123, 22.1140]]))
    # agent = LinearFeedbackPolicy(env, gain=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]))
    # agent = HeadTrackLinearFeedback(env, gain=np.array([[100, 10], [100, 10]]))
    render = "rgb_array"
    if save_video is None:
        render = None
    obs, acts, rews, imgs, ifs = exec_policy(env, agent,
                                             render=render, deterministic=True,
                                             repeat_num=len(load_kwargs['env_kwargs']['trials']),
                                             # infos=['torque'])
                                             infos=['torque', 'passive_torque', 'comx', 'comy', 'ptb_acc'])
    obs = np.array(obs)
    # acts = np.array(acts) * np.array([100, 150])
    tqs = np.array(ifs['torque'])
    for i in range(tqs.shape[0]):
        print(tqs[i].max(axis=0))
    ptqs = ifs['passive_torque']
    comxs = ifs['comx']
    comys = ifs['comy']
    ptb_forces = ifs['ptb_acc']
    # headxs = ifs['headx']

    title = name_tail[name_tail.rfind('/') + 1:]
    fig = draw_time_trajs_with_humandata(load_kwargs['env_kwargs']['trials'], obs, tqs, loaded_result['states'], loaded_result['torques'], title=title)
    # dt = env.get_attr("dt")[0]
    # fig = draw_time_trajs(obs, tqs, x=np.arange(0, tqs.shape[1])*1/120)
    # _ = draw_passive_torque(load_kwargs['env_kwargs']['trials'], tqs, ptqs, title=title)
    # fig = draw_versus_graph(load_kwargs['trials'], [comxs, obs[:,:,0], tqs[:, :, 0]], [comys, obs[:,:,1], tqs[:, :, 1]])
    # _ = draw_versus_graph(load_kwargs['env_kwargs']['trials'], [[np.arange(len(ptb_forces[i])) for i in range(len(ptb_forces))]], [ptb_forces])
    # fig = draw_versus_graph(load_kwargs['trials'], [len(headxs)*[np.arange(360)], len(obs)*[np.arange(360)], tqs[:, :, 0]], [headxs, np.array(obs)[:,:-1,4], tqs[:, :, 1]])

    if save_video is not None:
        video_record(imgs, f"videos/{save_video}.mp4", env.get_attr("dt")[0])
