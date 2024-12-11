import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import io

from common.sb3.util import make_env
from common.analyzer import video_record


def record_video():
    imgs = []
    env.reset()
    for state in states[31]:
        env.set_state(state[:2], state[2:])
        img = env.render("rgb_array")
        for _ in range(10):
            imgs.append(img)
    video_record(imgs, "./HumanSimVideo.mp4", env.dt)


def draw_figure():
    for _ in range(10):
        env.reset()
        plt.plot(env.ptb_acc)
    plt.show()


if __name__ == '__main__':
    subpath = os.path.join("IDP", "sub03", "sub03")
    states = [None for _ in range(35)]
    for i in range(31, 36):
        humanData = io.loadmat(subpath + f"i{i}.mat")
        states[i - 1] = humanData['state']
        bsp = humanData['bsp']
    env = make_env("IDPPD_MinEffort-v0", bsp=bsp, humanStates=states, ankle_max=100)
    record_video()
