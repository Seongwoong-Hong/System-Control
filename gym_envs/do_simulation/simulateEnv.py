import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from scipy.signal import find_peaks

from algos.torch.OptCont import LinearFeedbackPolicy
from common.sb3.util import make_env
from common.analyzer import video_record


def record_video():
    imgs = []
    env.reset()
    a = env.action_space.sample()
    # env.set_state(np.array([0.1]), np.array([0]))
    imgs.append(env.render("rgb_array"))
    done = False
    while not done:
        _, _, done, _ = env.step(np.ones(a.shape))
        imgs.append(env.render("rgb_array"))
    video_record(imgs, "./test.mp4", env.dt)

def step_env():
    ob = env.reset()
    obs = [ob]
    tqs = []
    for _ in range(360):
        act, _ = agent.predict(ob.astype(np.float32), deterministic=True)
        ob, _, done, info = env.step(act)
        # obs.append(np.rad2deg(info['prev_ob'].flatten()))
        obs.append(ob)
        tqs.append(info['torque'])
        if done:
            break
    return np.array(obs), np.array(tqs)

def cal_correlation():
    obs, tqs = step_env()
    Robs = np.array(obs)[-600:, :]
    R = (np.sum((Robs[:, 0] - np.mean(Robs[:, 0]))*(Robs[:, 1] - np.mean(Robs[:, 1]))) /
         (np.sqrt(np.sum((Robs[:, 0] - np.mean(Robs[:, 0]))**2))*np.sqrt(np.sum((Robs[:, 1] - np.mean(Robs[:, 1]))**2))))
    pks, _ = find_peaks(Robs[:, 0])
    peak1 = pks[0]
    pks, _ = find_peaks(Robs[:, 1])
    if peak1 < pks[0]:
        peak2 = pks[0]
    else:
        peak2 = pks[1]

    ph = frq * 360 / 120 * np.abs(peak2 - peak1)
    return R, ph

def draw_figure():
    fig = plt.figure(figsize=[8.0, 10.0])
    obs, tqs = step_env()
    x = np.arange(obs.shape[0] - 1)
    rT, cT = 4, max(round(obs.shape[-1] / 2), tqs.shape[-1])
    for i in range(rT * cT):
        fig.add_subplot(rT, cT, i + 1)
    for i in range(obs.shape[-1]):
        fig.axes[i].plot(x, np.rad2deg(obs[:-1, i]))
    for i in range(tqs.shape[-1]):
        fig.axes[obs.shape[-1] + i].plot(x, tqs[:, i])
    for _ in range(6):
        obs, tqs = step_env()
        x = np.arange(obs.shape[0] - 1)
        for i in range(obs.shape[-1]):
            fig.axes[i].plot(x, np.rad2deg(obs[:-1, i]))
        for i in range(tqs.shape[-1]):
            fig.axes[obs.shape[-1] + i].plot(x, tqs[:, i])
    fig.tight_layout()
    fig.show()
    return fig
    # data = {
    #     'hob': humanData['state'],
    #     'htq': humanData['tq'],
    #     'ob': obs,
    #     'tq': tqs,
    # }
    # io.savemat(f"ank.mat", data)


if __name__ == '__main__':
    subpath = os.path.join("..",  "..", "demos", "IDP", "sub10", "sub10")
    states = [None for _ in range(35)]
    for i in [1, 6, 11, 16, 21, 26, 31]:
        humanData = io.loadmat(subpath + f"i{i}.mat")
        states[i - 1] = humanData['state']
        states[i - 1][:] = 0.0
        bsp = humanData['bsp']
    Rs, phs = [], []
    env = make_env(
        "IDP_MinEffort-v0",
        bsp = bsp,
        ankle_torque_max=90,
        delay=True,
        ankle_limit="satu",
        humanStates=states,
    )
    # agent = LinearFeedbackPolicy(env, gain=np.array([[580.4426, 59.0801, 66.9362, 98.6479], [128.4063, 119.9887, 9.5562, 28.1239]]))
    agent = LinearFeedbackPolicy(env, gain=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]))
    # agent = LinearFeedbackPolicy(env, gain=2* np.array([[256.9201, 283.4496, 110.5109, 60.0833], [-22.1334, 188.7776, 30.5123, 22.1140]]))
    draw_figure()
    # frqs = np.arange(1.0, 10, 1.0)
    # for frq in frqs:
    #     env = make_env("IDP_SinPtb-v2",
    #                    bsp=None,
    #                    humanStates=states,
    #                    ankle_torque_max=500,
    #                    ptb_act_time=1/3,
    #                    stiffness=[1000, 2000],
    #                    damping=[50, 50],
    #                    # stiffness=[0, 0],
    #                    # damping=[0, 0],
    #                    delay=False,
    #                    delayed_time=0.1,
    #                    ankle_limit='soft',
    #                    frq_range=[frq],
    #                    )
    #     agent = LinearFeedbackPolicy(env, gain=np.array(
    #         [[649.5891, 291.7391, 90.2365, 94.8697], [-9.4614, 229.4411, 6.2950, 36.9884]]))
    #     draw_figure()
    #     R, ph = cal_correlation()
    #     Rs.append(R)
    #     phs.append(ph)
    # fig, ax1 = plt.subplots()
    # ax1.plot(frqs, Rs)
    # ax1.set_ylim(-1.0, 1.0)
    # ax2 = ax1.twinx()
    # ax2.plot(frqs, phs)
    # ax2.set_ylim(0, 180)
    # plt.show()

    ### ankle strategy: stiffness=[600, 500], damping = [400, 250]
    ### hip strategy: stiffness=[1000, 150], damping=[400, 60]
    # agent = LinearFeedbackPolicy(env, gain=np.array([[649.5891, 291.7391, 90.2365, 94.8697], [-9.4614, 229.4411, 6.2950, 36.9884]]))
    # agent = LinearFeedbackPolicy(env, gain=np.array([[349.5891, 0, 30.2365, 0], [0, 129.4411, 0, 16.9884]]))
